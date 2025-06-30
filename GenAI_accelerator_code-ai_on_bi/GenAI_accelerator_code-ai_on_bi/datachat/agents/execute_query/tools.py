import io
import os
from tempfile import NamedTemporaryFile
from typing import Annotated
from sqlalchemy.exc import ProgrammingError  
from sqlalchemy import create_engine

import pandas as pd
import pymssql
import snowflake.connector
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from snowflake.connector.errors import ProgrammingError


class DataFrameInfo(BaseModel):
    info: str = Field(description="Brief info of the dataframe")
    path: str = Field(description="Path where the dataframe is stored")
    columns: dict = Field(description="Dataframe columns and datatype")


@tool
def query_db(
    # Injectstate denotes that the parameter is obtained from the state. This should avoid adding extra phrase to the query
    # Refer: https://langchain-ai.github.io/langgraph/how-tos/pass-run-time-values-to-tools/
    query: Annotated[str, InjectedState("sql_query")]
):
    """Executes a SQL query in Snowflake and returns the results as a Pandas DataFrame.

    Args:
        query: The SQL query to execute

    Returns:
        A Pandas Dataframe
    """
    print("EXECUTING SQL QUERY===>", query)
    query = query.strip().strip("`").strip('"').strip()
    query = query.replace("query = ", "").strip('"')
    server = os.environ["AZURE_SQL_SERVER_NAME"]
    database = os.environ["AZURE_SQL_SERVER_DATABASE"]
    username = os.environ["AZURE_SQL_SERVER_USER"]
    password = os.environ["AZURE_SQL_SERVER_PASSWORD"]
    conn = pymssql.connect(server=server, user=username, password=password, database=database)

    try:
        cursor = conn.cursor()
        cursor.execute(query)
    except Exception as e:
        print(f"Query ```{query}``` is invalid", e)
        return e

    df = pd.read_sql(query, conn)

    buffer = io.StringIO()
    df.info(buf=buffer, memory_usage=False, verbose=True)
    df_info = buffer.getvalue()
    buffer.close()

    df_temp_file = NamedTemporaryFile(
        delete=False
    )  # Do not close the file right away. This file is needed to read the df in the chat UI
    df.to_pickle(df_temp_file.name)
    df_columns = df.dtypes.to_dict()
    return DataFrameInfo(info=df_info, path=df_temp_file.name, columns=df_columns)
