import os
import json
from langchain.chains import RetrievalQA

from datachat.agents.helpers import get_llm
from datachat.utils.vectordb import init_vector_store

LANCE_URI = os.environ["LANCE_URI"]
TABLE_NAME = os.environ["VECTORDB_TABLE_NAME"]

llm = get_llm()
vector_store = init_vector_store(LANCE_URI, TABLE_NAME)

retriever = vector_store.as_retriever(score_threshold= 0.3,search_kwargs={"k": 6})

prompt = f"""You are an expert database assistant. Based on the following question, retrieve all relevant table and column metadata to formulate a SQL query.
                
                Question: Give me product names which have maximum sales amount in last year along with maximum sales amount.

                Instructions:
                - Only return table and column information from the given metadata.
                - Do not assume column names. If a required column is missing, find the closest matching column from the metadata.
                - Ensure all necessary tables and columns are included for query formulation.
                - Provide the output in the following JSON format, parseable by `json.loads()`:
                - Avoid adding extra explanations or comments.
                - For date/month/year related questions, retrieve the columns which related to date.
                Expected Output (Example):
                [
                {{
                    "column": "column_name",
                    "datatype": "column_datatype",
                    "description": "column_description",
                    "foreign_key_ref": "NULL or Table_name.Column_name",
                    "primary_key": true,
                    "table": "table_name"
                }}
                ]
            """

retrieval_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    verbose=True
)
response = retrieval_chain.invoke(prompt)

print(json.loads(response['result']))