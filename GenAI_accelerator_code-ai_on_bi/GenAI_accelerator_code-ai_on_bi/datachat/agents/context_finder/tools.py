import ast
import json
import os
import lancedb
from typing import Annotated
import streamlit as st
import pandas as pd

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langchain.chains import RetrievalQA
from langchain_openai import AzureChatOpenAI
from langgraph.types import interrupt

from datachat.agents.helpers import get_llm
from datachat.utils.vectordb import init_vector_store


import os
import lancedb
import pyarrow as pa
from langchain_openai import AzureOpenAIEmbeddings

# === CONFIG ===
VECTOR_DIM = 1536
DB_PATH = "lance_db"
TABLE_NAME = "suits"
MODEL = "text-embedding-ada-002"  # your embedding model name
DEPLOYMENT_NAME = "llm-acelerator-embedding"


embeddings = AzureOpenAIEmbeddings(
    azure_deployment="llm-acelerator-embedding",
    model="text-embedding-ada-002",
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


LANCE_URI = os.environ["LANCE_URI"]
TABLE_NAME = os.environ["VECTORDB_TABLE_NAME"]

llm = get_llm()


llm1 = AzureChatOpenAI(
            api_version="2023-03-15-preview",
            api_key="",
            azure_endpoint="",
            azure_deployment="gpt-4o-mini",
            model="gpt-4o-mini",
            openai_api_type="azure",
            temperature=0,
        )


def search_similar_inputs(query, k=3):
    embedding = generate_embeddings(query)
    results = table.search(embedding).limit(k).to_pandas()  # Use to_pandas instead of to_df
    print("Returned columns:", results.columns.tolist())    # Debug: print column names

    # Rename _distance to score if it exists
    if "_distance" in results.columns:
        results = results.rename(columns={"_distance": "score"})

    return results

@tool
def context_finder_tool(question: Annotated[str, InjectedState("input")]):
    """Generate and returns a list of documents relevant to the user question.
    Args:
        question: User Question in Natural Language.

    Returns:
        A List of Dictionaries[Dict].
    """
    # docs_with_scores = vector_store.similarity_search_with_relevance_scores(
    #     query=question, kwargs={"score_threshold": 0.5}
    # )
    print ("----++++++++++")
    # st.write(st.session_state.selected_function)
    vector_store = init_vector_store(LANCE_URI, st.session_state.selected_function)
    column_vector_store= init_vector_store(LANCE_URI, TABLE_NAME)
    retriever = vector_store.as_retriever(score_threshold= 0.1)
    retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
    )
    column_retriever = column_vector_store.as_retriever(score_threshold= 0.1)
    column_retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=column_retriever,
    )
    print("question")
    print(question)
    prompt = f"""You are an expert database assistant. Based on the following question, retrieve all relevant table  metadata to formulate a SQL query.
            
            Question: {question}

            Instructions:
            - Only return table  information from the given metadata.
            - For  financial metrics Search through the table descriptions to find tables that mention concepts like `sales`, `revenue`, `payment`, `transactions`, `time`, `date`,etc.
            - return all the tables infromation to formulate a query.if there is no such table give the most relevant table.
            - Provide the output in the following JSON format, parseable by `json.loads()`:
            - Avoid adding extra explanations or comments.
            - Give the complete description.
            - If the question is asked specific for location retrieve the tables related to location and address as well.
            Expected Output (Example):
            [
            {{
                "table": "table_name",
                "description": "table_description"
            }},
            ....
            ]
        """
    
    column_prompt = f"""You are an expert database assistant. Based on the following question, retrieve all relevant column metadata needed to generate a SQL query for the following question:
                
                Question: {question}

                Instructions:
                - Only return column information from the given metadata, judging them by the column description.
                - Retrieve all relevant columns needed to formulate a SQL query.
                - For **financial metrics**, search through the column descriptions to find columns that mention concepts like `sales`, `revenue`, `payment`, `transactions`, `time`, `date`, `amount`, `cost`, `profit`, etc.
                - For **location-related queries**, look for columns that mention concepts like `country`, `region`, `location`, `warehouse`, etc.
                - Do not assume column names; instead, focus on column descriptions for relevance.
                - Ensure all necessary tables and columns are included to formulate an accurate SQL query.
                - Avoid adding extra explanations or comments.

                Expected Output:
                [
                {{
                    "column": "column_name",
                    "datatype": "column_datatype",
                    "description": "column_description",
                    "foreign_key_ref": "NULL or Table_name.Column_name",
                    "primary_key": true,
                    "table": "table_name"
                }},
                {{
                    "column": "column_name",
                    "datatype": "column_datatype",
                    "description": "column_description",
                    "foreign_key_ref": "NULL or Table_name.Column_name",
                    "primary_key": true,
                    "table": "table_name"
                }},
                ....
                ]
            """


    print(question)

    response = retrieval_chain.invoke(prompt)
    try:
        res = json.loads(response['result'])
    except json.JSONDecodeError:
        response['result'] = response['result'].strip('`').replace("json","")
        res = json.loads(response['result'])
    print(res)

    column_response = column_retrieval_chain.invoke(column_prompt)
    print(column_response)
    try:
        column_res = json.loads(column_response['result'])
    except json.JSONDecodeError:
        column_response['result'] = column_response['result'].strip('`').replace("json","")
        column_res = json.loads(column_response['result'])
    print("column_res")
    print(column_res)

    try:
        table_list = res # Convert response to dictionary
        table_names = {table["table"] for table in table_list}  # Use a set for faster lookup
    except json.JSONDecodeError:
        print("Error parsing table response JSON")
        table_names = set()
    print(table_names)

    # Initialize LanceDB connection
    db = lancedb.connect(LANCE_URI)

    # Open the column metadata table
    column_table = db.open_table("suits")

    # Convert to Pandas DataFrame
    df = column_table.to_pandas()

    # Print column names for debugging
    print("Available Columns:", df.columns.tolist())

    # Ensure 'metadata' column exists
    if "metadata" not in df.columns:
        raise ValueError("No 'metadata' column found in the vector store!")

    # Check if metadata needs parsing
    def parse_metadata(meta):
        if isinstance(meta, str):  # Parse only if it's a JSON string
            return json.loads(meta)
        return meta  # If it's already a dict, return as-is

    # Apply parsing
    df["metadata"] = df["metadata"].apply(parse_metadata)

    # Extract table names from metadata
    df["table"] = df["metadata"].apply(lambda x: x.get("table", None))  # Extract 'table' key

    df["table"] = df["table"].astype(str).str.strip().str.upper()

    # Ensure table_names set contains only properly formatted strings
    table_names = {str(name).strip().upper() for name in table_names}
    
    # Filter only rows where 'table' matches the given names
    filtered_columns = df[df["table"].isin(table_names)]

    # Drop unnecessary columns (keep metadata)
    filtered_columns = filtered_columns[["table", "metadata"]]

    # Convert to JSON format
    filtered_columns_json = filtered_columns.to_dict(orient="records")

    # Extract only metadata values from filtered results
    final_column_metadata = [row["metadata"] for row in filtered_columns_json]

    # Print structured JSON output
    final_res=json.dumps(final_column_metadata)

    final_column_metadata = final_column_metadata + column_res
    #print(final_column_metadata)
    if len(st.session_state.add_docs) > 0:
        print(final_column_metadata)
        add_docs = st.session_state.get("add_docs")
        final_column_metadata+=add_docs.to_dict(orient="records")
        print(final_column_metadata)
    context_valid_prompt=f"""
    You are given a natural language question and a set of table metadata. 
    Determine if it is possible to formulate a valid SQL query that answers the question using only the information available in the metadata (e.g., table names, column names, data types, and column descriptions).
    Do not infer or assume any information beyond what is explicitly stated in the metadata.
    You should not assume information that is not present. However:
        - If a datetime column like `transaction_date`, `created_at`, or similar is present, you may use standard SQL functions to extract year, month, or quarter.
        - If a datetime column like `transaction_date`, `created_at`, or similar is present, you may use standard SQL functions to extract year, month, or quarter.
        - If a question asks for a breakdown (e.g., revenue by category), and there are columns that *could* reasonably serve as group-by dimensions (e.g., product_id, item_type, product_name), you may use those, even if the exact words "product line" or "service type" are not in the metadata.
        - If numeric columns like `total_price`, `sales_amount`, or `revenue` are present, you may use them to calculate revenue or totals.   
        - Do not interpret phrases like “best practices” or “methodologies” unless the question *explicitly* asks for them.
        - If SQL can be used to show a month-over-month breakdown of numeric values over time (e.g., revenue), then the answer should be **Yes**.
        - If a question involves customer behavior , and the metadata includes relevant fields like customer IDs and purchase timestamps, then it **can** be answered with SQL and the answer should be **Yes**.
        - you don't need any information about specific methods, metrics, or analysis techniques you only consider required tables and columns are present or not.
        - You should **not** reject a question just because it mentions business concepts like “retention” or “repeat behavior,” if these can be computed using joins, aggregations, or SQL logic based on the available columns.
        - If a question asks for insights that can be derived through standard SQL logic (such as classifying users, calculating frequencies, or segmenting based on counts or timestamps), and the required raw data (e.g., user IDs, transaction IDs, timestamps) is present in the metadata, then the answer should be **Yes**. You do not need explicit metrics, documents, or pre-defined models for such logic.
        - Do not interpret phrases like “best practices” or “methodologies” unless the question *explicitly* asks for them.
        - If the question involves customer behavior (e.g., repeat purchases, customer retention, or behavior over time), and the metadata includes relevant columns such as `CUSTOMER_ID`, `PURCHASE_DATE`, or similar, then the answer should be **Yes**, as these can be used to derive insights into customer behavior using SQL logic.
        - only consider table metadata.
        - SQL can calculate year-over-year trends, state-wise breakdowns, and more using aggregations and groupings.
        - additional context or specific relationships outlined in the metadata are not required ,
        - Trends or comparisons over time can often be inferred if columns like sale_date, amount, state, etc. are present.
        - Explain your reasoning in simple terms without using ant terms like metadata(example:The data doesn’t include any details about employees or their salaries.).

    Note:
        - Give no only if its not at all possible to solve the question with the metadata available.
    Input:
        Question: {question}

        Metadata: {final_column_metadata}

    expected output:
        {{
        "answer": "Yes" or "No",
        "explanation": "Explain your reasoning in simple terms."
        }}
    
    

    """

    context_valid=llm.invoke(context_valid_prompt).content
    context_valid_json = json.loads(context_valid)
    print("%%%%%%%5%%555%%%%%")
    print(question)
    results = search_similar_inputs(st.session_state.user_input)
    print(results[["score",'user_input', 'sql_query']])
    st.session_state.similar_inputs=results[["score",'user_input', 'sql_query']]
    if (results["score"] < 0.28).any() or context_valid_json["answer"]=="Yes":
        print(context_valid)
        st.session_state.similar_inputs=results[["score",'user_input', 'sql_query']]
        st.session_state.state_docs=final_column_metadata
        return final_column_metadata
    else:
        st.session_state.context_docs=context_valid
        st.session_state.awaiting_human=True
        st.session_state.docs=final_column_metadata
        print(context_valid)
        st.session_state.context_interupt=1
        return interrupt("human_input")

