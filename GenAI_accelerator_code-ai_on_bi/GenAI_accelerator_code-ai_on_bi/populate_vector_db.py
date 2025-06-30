import json
import logging
import os

import pandas as pd
from langchain_community.vectorstores.lancedb import LanceDB
from langchain_openai.embeddings.azure import AzureOpenAIEmbeddings

logger = logging.getLogger(__file__)

LANCE_URI = os.environ["LANCE_URI"]
TABLE_NAME = os.environ["VECTORDB_TABLE_NAME"]


def generate_column_docs(meta_df):
    docs = []
    doc_metadata = []
    for _, row in meta_df.iterrows():
        data = {
            "table": row["table"],
            "column": row["column"],
            "datatype": row["column_datatype"],
            "description": row["column_description"],
        }
        doc = list(data.values())

        if row["primary_key"]:
            data["primary_key"] = row["primary_key"]
            doc.append("PRIMARY KEY")
        if not pd.isnull(row["foreign_key_ref"]):
            data["foreign_key_ref"] = row["foreign_key_ref"]
            doc.append(f"FOREIGN KEY REFERENCES {row['foreign_key_ref']}")

        doc_metadata.append(data)
        docs.append(" ".join(doc))

    return docs, doc_metadata


def generate_table_docs(meta_df):
    docs = []
    doc_metadata = []
    for _, row in meta_df.iterrows():
        data = {
            "table": row["table"],
            "description": row["table_description"],
        }
        doc = list(data.values())
        docs.append(" ".join(doc))
    return docs, doc_metadata

def populate_vector_store(metadata, lance_uri,TABLE_NAME):
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.environ["EMBEDDING_AZURE_ENDPOINT"],
        azure_deployment=os.environ["EMBEDDING_AZURE_DEPLOYMENT"],
        api_version=os.environ["EMBEDDING_AZURE_API_VERSION"],
        openai_api_type="azure",
        model=os.environ["EMBEDDING_AZURE_MODEL"],
        api_key=os.environ["EMBEDDING_AZURE_API_KEY"],
    )

    schema_docs, doc_metadata = generate_column_docs(metadata)
    vector_store = LanceDB(uri=lance_uri, embedding=embeddings, table_name=TABLE_NAME)

    _ = vector_store.add_texts(texts=schema_docs, metadatas=doc_metadata)

    return vector_store

def populate_table_vector_store(metadata, lance_uri,TABLE_NAME):
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.environ["EMBEDDING_AZURE_ENDPOINT"],
        azure_deployment=os.environ["EMBEDDING_AZURE_DEPLOYMENT"],
        api_version=os.environ["EMBEDDING_AZURE_API_VERSION"],
        openai_api_type="azure",
        model=os.environ["EMBEDDING_AZURE_MODEL"],
        api_key=os.environ["EMBEDDING_AZURE_API_KEY"],
    )

    schema_docs, doc_metadata = generate_table_docs(metadata)
    vector_store = LanceDB(uri=lance_uri, embedding=embeddings, table_name=TABLE_NAME)

    _ = vector_store.add_texts(texts=schema_docs, metadatas=doc_metadata)

    return vector_store


def create_function_specific_lance_tables(function_table_df, table_metadata_df, lance_uri):
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.environ["EMBEDDING_AZURE_ENDPOINT"],
        azure_deployment=os.environ["EMBEDDING_AZURE_DEPLOYMENT"],
        api_version=os.environ["EMBEDDING_AZURE_API_VERSION"],
        openai_api_type="azure",
        model=os.environ["EMBEDDING_AZURE_MODEL"],
        api_key=os.environ["EMBEDDING_AZURE_API_KEY"],
    )

    for function in function_table_df["function"].unique():
        # Get list of tables for the current function
        tables_for_function = function_table_df[function_table_df["function"] == function]["table"].unique()

        # Filter metadata for those tables
        filtered_metadata = table_metadata_df[table_metadata_df["table"].isin(tables_for_function)]

        # Generate LanceDB docs and metadata
        docs, doc_metadata = generate_table_docs(filtered_metadata)

        # Create Lance table (name it using function, replacing spaces with underscores)
        table_name = function.lower().replace(" ", "_")

        vector_store = LanceDB(uri=lance_uri, embedding=embeddings, table_name=table_name)
        if docs:
            _ = vector_store.add_texts(texts=docs, metadatas=doc_metadata)

        print(f"✅ Created LanceDB table: {table_name} with {len(docs)} entries.")



if __name__ == "__main__":
    logger.info("Reading CSV file as pandas")
    metadata = pd.read_csv("data/snowflake_data.csv")

    logger.info("Adding dataframe info to vector db")
    vector_store = populate_vector_store(metadata, LANCE_URI,TABLE_NAME)

    metadata = pd.read_csv("data/table_metadata.csv")

    logger.info("Adding dataframe info to vector db")
    vector_store = populate_table_vector_store(metadata, LANCE_URI,"table2")


    import pandas as pd

    function_table_df = pd.DataFrame([
        ("Sales_Revenue_Performance", "FactInternetSales"),
        ("Sales_Revenue_Performance", "DimDate"),
        ("Sales_Revenue_Performance", "FactResellerSales"),
        ("Sales_Revenue_Performance", "DimSalesTerritory"),
        ("Sales_Revenue_Performance", "DimCustomer"),
        ("Sales_Revenue_Performance", "DimProduct"),
        ("Customer_Insights", "DimCustomer"),
        ("Customer_Insights", "FactInternetSales"),
        ("Customer_Insights", "DimGeography"),
        ("Product_Category_Analytics", "DimProduct"),
        ("Product_Category_Analytics", "DimProductSubcategory"),
        ("Product_Category_Analytics", "DimProductCategory"),
        ("Product_Category_Analytics", "FactInternetSales"),
        ("Product_Category_Analytics", "FactProductInventory"),
    ], columns=["function", "table"])


    logger.info("Creating LanceDB tables by function")
    create_function_specific_lance_tables(function_table_df, metadata, LANCE_URI)


    logger.info("Vector db initialized")
