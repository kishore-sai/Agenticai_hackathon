import json
import os
from itertools import count
from typing import Annotated, Any, Dict, Optional
import streamlit as st
from langgraph.types import interrupt

import os
import lancedb
import pyarrow as pa
from langchain_openai import AzureOpenAIEmbeddings

import sqlglot as sg
import sqlglot.expressions as exp
from langchain.chains.combine_documents.base import (
    DEFAULT_DOCUMENT_PROMPT,
    DEFAULT_DOCUMENT_SEPARATOR,
    DOCUMENTS_KEY,
    _validate_prompt,
)
from langchain.chains.retrieval import create_retrieval_chain
from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import LanguageModelLike
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.prompts import BasePromptTemplate, PromptTemplate, format_document
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field

from datachat.agents.helpers import get_llm
from datachat.utils.vectordb import init_vector_store

LANCE_URI = os.environ["LANCE_URI"]
TABLE_NAME = os.environ["VECTORDB_TABLE_NAME"]

llm = get_llm()
vector_store = init_vector_store(LANCE_URI, TABLE_NAME)


# === CONFIG ===
VECTOR_DIM = 1536
DB_PATH = "lance_db"
TABLE_NAME = "suits"
MODEL = "text-embedding-ada-002"  # your embedding model name
DEPLOYMENT_NAME = "llm-acelerator-embedding"


azure_endpoint=os.environ["EMBEDDING_AZURE_ENDPOINT"]
azure_deployment=os.environ["EMBEDDING_AZURE_DEPLOYMENT"]
api_version=os.environ["EMBEDDING_AZURE_API_VERSION"]
openai_api_type="azure"
model=os.environ["EMBEDDING_AZURE_MODEL"]
api_key=os.environ["EMBEDDING_AZURE_API_KEY"]
chunk_size=2048

MODEL = "text-embedding-ada-002"
embeddings = AzureOpenAIEmbeddings(
    azure_deployment="llm-acelerator-embedding",
    model=MODEL,
    chunk_size=2048,
    azure_endpoint=os.environ["EMBEDDING_AZURE_ENDPOINT"],
)

def generate_embeddings(text):
    return embeddings.embed_query(text)

# === LANCEDB SETUP ===
db = lancedb.connect(DB_PATH)

schema = pa.schema([
    pa.field("user_input", pa.string()),
    pa.field("sql_query", pa.string()),
    pa.field("embedding", pa.list_(pa.float32(), VECTOR_DIM)),
])

if TABLE_NAME not in db.table_names():
    table = db.create_table(TABLE_NAME, schema=schema)
else:
    table = db.open_table(TABLE_NAME)



def search_similar_inputs(query, k=3):
    embedding = generate_embeddings(query)
    results = table.search(embedding).limit(k).to_df()
    return results

def create_table_ddl_chain(
    llm: LanguageModelLike,
    prompt: BasePromptTemplate,
    *,
    output_parser: Optional[BaseOutputParser] = None,
    document_prompt: Optional[BasePromptTemplate] = None,
    document_separator: str = DEFAULT_DOCUMENT_SEPARATOR,
    document_variable_name: str = DOCUMENTS_KEY,
) -> Runnable[Dict[str, Any], Any]:
    _validate_prompt(prompt, document_variable_name)
    _document_prompt = document_prompt or DEFAULT_DOCUMENT_PROMPT
    _output_parser = output_parser or StrOutputParser()

    constraint_count = count()

    def generate_col_defn(col):
        col_defn = [
            " ",
            col["column"]
            #col["datatype"],
        ]
        #if col.get("primary_key") != False:
        #    col_defn.append("CONSTRAINT PRIMARY KEY")
        """
        if col.get("foreign_key_ref") and col.get("foreign_key_ref") != 'NULL':
            col_defn.extend(
                [
                    "CONSTRAINT",
                    f"c_{next(constraint_count)}",
                    "FORIEGN_KEY",
                    "REFERENCES",
                    col.get("foreign_key_ref").split(".")[0],  # Table name
                    col.get("foreign_key_ref").split(".")[1],  # Column name
                ]
            )
        """
        col_defn.extend(
            [
                "COMMENT",
                "'" + col["description"] + "'",
            ]
        )

        return " ".join(col_defn)

    def generate_table_ddl(tbl, cols_defn):
        all_cols_ddl = ",\n".join(cols_defn)
        return f"CREATE TABLE {tbl} (\n{all_cols_ddl}\n)"

    def format_docs(input_docs):
        # print("INPUTS:", inputs)

        table_cols = {}
        for doc in input_docs[document_variable_name]:
            col = doc
            try:
                if col["table"]:
                    tbl = col["table"]
            except KeyError:
                st.write(col)
            if tbl not in table_cols:
                table_cols[tbl] = []

            print("Generating defn for column", col["column"], "in table", tbl)
            col_defn = generate_col_defn(col)
            table_cols[tbl].append(col_defn)

        all_ddls = []
        for k, v in table_cols.items():
            all_ddls.append(generate_table_ddl(k, v))
        return "\n".join(all_ddls)

    return (
        RunnablePassthrough.assign(**{document_variable_name: format_docs}).with_config(
            run_name="format_inputs"
        )
        | prompt
        | llm
        | _output_parser
    ).with_config(run_name="table_ddl_chain")



class QueryAndExplanation(BaseModel):
    query: str = Field(description="A snowflake dialect valid SQL query for answering the user question")
    explanation: str = Field(
        description="Explanation for how the sql query answers the user question"
    )

    @property
    def as_chat_message(self):
        return f"""{self.explanation}
        ```
        {self.query}
        ```
        """


class QueryInput(BaseModel):
    question: str = Field(description="Input user question")
    docs: list = Field(
        description="Context Documents generated for user question"
    )

@tool(args_schema=QueryInput)
def generate_query_for_question(question: str, docs: list):
    """Generate a SQL Query in sql server(T-SQL) dialect based on the given list of context documents for user question. Doesn't have permission to execute the query.

    Args:
        question: User Question in Natural Language.
        docs: Context Documents generated for user question
    
    Example Input:
        question: How many store do we have in each state?
        docs: [{'column': 'STORE_ID', 'datatype': 'NUMBER(38,0)', 'description': 'Unique identifier for each store', 'foreign_key_ref': Null, 'primary_key': True, 'table': 'STORE_DATA'},{'column': 'STATE', 'datatype': 'VARCHAR', 'description': 'State where the store is located', 'foreign_key_ref': Null, 'primary_key': False, 'table': 'STORE_DATA'}, ....]

    Returns:
        A QueryAndExplanation object.
    """
    print("Inside Generate query function")
    parser = PydanticOutputParser(pydantic_object=QueryAndExplanation)
    response_parser = OutputFixingParser.from_llm(llm=llm, parser=parser, max_retries=3)
    prompt = PromptTemplate(
    template="\n".join(
        [
            "You are a skilled BI assistant that converts natural language business questions into efficient and accurate SQL Server queries.",
            "",
            "Question: {input}",
            "",
            "If the question is exactly the same as one from the hints below, or very similar, return the same SQL as in the hint.",
            "Previous Questions and Answers (hints):",
            "{hints}",
            "",
            "You can ONLY use the following context (tables, columns, and their descriptions):",
            "{context}",
            "",
            "{format_instructions}",
            "",
            "Guidelines:",
            "1. Only return the final SQL query and nothing else. Do not include explanations.",
            "2. The result should be valid T-SQL (SQL Server compatible).",
            "3. Use human-readable column names (like Name, Region, Category) instead of keys (like CustomerKey, ProductKey) unless the question explicitly asks for IDs.",
            "4. Refer to column descriptions to choose the most appropriate columns for the question.",
            "5. For segmentation or classification (e.g., new vs repeat customers), first compute metrics per individual (e.g., per customer), then group and count or summarize.",
            "6. The query result should be directly usable for BI visualizations — this means including grouping/categorical columns (e.g., Region, Month, Product) alongside aggregated metrics.",
            "7. When aggregating metrics like sales or revenue across multiple channels or sources, combine results from all relevant tables using UNION or UNION ALL to produce a unified summary (e.g., total yearly sales from different sales fact tables).",
            "8. Avoid aliasing inside GROUP BY. Instead, group by original column references or expressions.",
            "9. If CASE statements or derived values are needed, wrap the logic in subqueries or CTEs.",
            "10. When combining multiple tables (e.g., facts and dimensions), ensure proper joins using relevant keys, and prefer displaying meaningful attributes from the dimension tables.",
            "11. Do not use window functions inside GROUP BY or WHERE — only use them in SELECT or ORDER BY clauses as per SQL Server constraints.",
            "12. Ensure the SQL output supports typical BI use cases such as trends, KPIs, breakdowns, comparisons, and segmentation.",
            "13. The output JSON must be valid and parsable using Python's json.loads().",
            "14. When performing segmentation or classification based on aggregates (e.g., customer type, product tier, user activity level), first aggregate the relevant metric at the appropriate entity level (e.g., per customer, per product), then apply the classification logic (e.g., with CASE) in a CTE or subquery, and finally summarize the results.",
            "15. Before referencing columns in a query, ensure their existence by validating against the provided table metadata. Use exact column names as listed, and leverage column descriptions to infer their purpose.",
            "16. Also make sure the column u'r using s present in the particular table u'r refering to",
            "17. When filtering by date on columns like DateKey or OrderDateKey, which are integer-based keys (e.g., YYYYMMDD), convert date literals to integers instead of using string date formats.",
            "18. Don't use any kind of datekey columns use only date columns.",
            "19. When giving name columns, if there are columns like first name and last name, combine them and give as a single column in the output.",
            "20. Do NOT nest aggregate functions (e.g., SUM, COUNT) inside window functions like LAG or LEAD. Instead, first compute the aggregated values using a subquery or CTE, and then apply the window functions over the aggregated results.",
            "21. If a question is identical or very similar to one in the hints section, return the same SQL query used in the hint. Do NOT attempt to rewrite or optimize it differently."
        ]
    ),

        input_variables=['context', 'hints', 'input'],
        partial_variables={"format_instructions": parser.get_format_instructions()},
        output_parser=parser,
    )

    rag_chain = create_table_ddl_chain(
        llm=llm, prompt=prompt, output_parser=response_parser
    )
    # docs= InjectedState("docs_with_scores")
    #qa_chain = (
    #    RunnablePassthrough.assign(context=(lambda x: docs)).assign(
    #        answer=rag_chain
    #    )
    #).with_config(run_name="retrieval_chain")
    results = search_similar_inputs(question)
    hint = "\n\n".join(
        f"Q: {row['user_input']}\nA: {row['sql_query']}"
        for _, row in results.iterrows()
    )
    qa_chain = (
        RunnablePassthrough.assign(context=(lambda x: docs))  # Injecting context from docs_with_scores
        .assign(hints=(lambda x: hint))  # Injecting hint
        .assign(answer=rag_chain)  # Assign the rag_chain for query generation
    ).with_config(run_name="retrieval_chain")
    llm_response = qa_chain.invoke({"input": question})
    if isinstance(llm_response["answer"], QueryAndExplanation):
        print("&&&&&&&&&&&&&&&&&&&&&")
        print("Question------")
        print(st.session_state.state_docs)
        if st.session_state.context_interupt==None:
            return llm_response["answer"]
        else:
            st.session_state.sql_query= llm_response["answer"].query
            st.session_state.explanation= llm_response["answer"].explanation
            st.session_state.context_docs=dict()
            print(st.session_state.sql_query)
            return interrupt("human_input")

    try:
        parsed_response = parser.parse(llm_response["answer"])
        print("hii")
    except OutputParserException as oe:
        print("Handled the OutputParserException using ast.literal_eval")
        import ast

        parsed_response = QueryAndExplanation.parse_obj(
            ast.literal_eval(llm_response["answer"])
        )
    return parsed_response


