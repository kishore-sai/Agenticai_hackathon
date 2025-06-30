import itertools
import json
import os
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import langchain
import openai
import outlines
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.combine_documents.base import (
    DEFAULT_DOCUMENT_PROMPT,
    DEFAULT_DOCUMENT_SEPARATOR,
    DOCUMENTS_KEY,
    BaseCombineDocumentsChain,
    _validate_prompt,
)
from langchain.chains.llm import LLMChain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import LanguageModelLike
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.prompts import (
    BasePromptTemplate,
    ChatPromptTemplate,
    PromptTemplate,
    format_document,
)
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_openai import AzureChatOpenAI

langchain.debug = os.environ.get("LANGCHAIN_DEBUG", False)

llm = AzureChatOpenAI(
    azure_endpoint=os.environ["LLM_AZURE_ENDPOINT"],
    azure_deployment=os.environ["LLM_AZURE_DEPLOYMENT"],
    api_version=os.environ["LLM_AZURE_API_VERSION"],
    openai_api_type="azure",
    model=os.environ["LLM_AZURE_MODEL"],
    verbose=True,
    temperature=0
)


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

    def generate_col_defn(col):
        return " ".join(
            [
                " ",
                col["column"],
                col["datatype"],
                "COMMENT",
                "'" + col["description"] + "'",
            ]
        )

    def generate_table_ddl(tbl, cols_defn):
        all_cols_ddl = ",\n".join(cols_defn)
        return f"CREATE TABLE {tbl} (\n{all_cols_ddl}\n)"

    def format_docs(input_docs):
        # print("INPUTS:", inputs)

        table_cols = {}
        for doc in input_docs[document_variable_name]:
            col = json.loads(format_document(doc, _document_prompt))
            tbl = col["table"]
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


class Graph(BaseModel):
    xlabel: str = Field(description="Column name to be placed in x-axis of the graph")
    ylabel: str = Field(description="Column name to be placed in y-axis of the graph")


class QueryAndDescription(BaseModel):
    query: str = Field(description="A valid SQL query for answering the user question")
    description: str = Field(description="Description of the executable sql query")

    @property
    def as_chat_message(self):
        return f"""{self.description}
        ```
        {self.query}
        ```
        """


def generate_query_for_question(question, vector_store):
    parser = PydanticOutputParser(pydantic_object=QueryAndDescription)
    prompt = PromptTemplate(
        template="\n".join(
            [
                "Generate a SQL for the following question.",
                "Question: {input}",
                "\n",
                "{format_instructions}",
                "Note: The output json should be parsable by Python's json.loads function.",
                "\n",
                "Relevant table columns:",
                "{context}",
            ]
        ),
        input_variables=["input", "context"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
        output_parser=parser,
    )

    rag_chain = create_table_ddl_chain(llm=llm, prompt=prompt)
    qa_chain = create_retrieval_chain(
        retriever=vector_store.as_retriever(), combine_docs_chain=rag_chain
    )

    llm_response = qa_chain.invoke({"input": question})
    print(llm_response)
    try:
        parsed_response = parser.parse(llm_response["answer"])
    except OutputParserException as oe:
        print("Handled the OutputParserException using ast.literal_eval")
        import ast

        parsed_response = QueryAndDescription.parse_obj(
            ast.literal_eval(llm_response["answer"])
        )
    return parsed_response


def choose_vizualization_type(question):
    vizualization_prompt = f"""You are a Data Vizualization expert. For the following question, choose the type of visualization for the pandas dataframe.

    Question: {question}
    """

    model = outlines.models.azure_openai(
        deployment_name=os.environ["LLM_AZURE_DEPLOYMENT"],
        api_version=os.environ["LLM_AZURE_API_VERSION"],
        model_name=os.environ["LLM_AZURE_MODEL"],
        azure_endpoint=os.environ["LLM_AZURE_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
    )
    gentr = outlines.generate.choice(
        model, ["line", "bar", "hist", "box", "density", "area", "pie", "scatter"]
    )
    plot_type = gentr(vizualization_prompt)
    print("LLM Output for plot:", plot_type)

    return plot_type


def choose_plot_dims(question, columns):

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a Data Vizualization expert.
        Given a list of columns and a question, extract the columns into x and y axis labels.
        {format_instructions}
        The output should be parseable by Python's json.loads() function.""",
            ),
            (
                "user",
                f"""Question: {question}
        Columns: {", ".join(columns)}""",
            ),
        ]
    )

    parser = PydanticOutputParser(pydantic_object=Graph)

    chain = prompt | llm | parser

    viz_labels = chain.invoke({"format_instructions": parser.get_format_instructions()})
    print("Output of choose_plot_dims:", viz_labels)
    return viz_labels
