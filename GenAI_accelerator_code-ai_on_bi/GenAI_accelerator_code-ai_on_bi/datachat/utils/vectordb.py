import os

from langchain_community.vectorstores.lancedb import LanceDB
from langchain_openai.embeddings.azure import AzureOpenAIEmbeddings


def init_vector_store(uri, table_name):
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.environ["EMBEDDING_AZURE_ENDPOINT"],
        azure_deployment=os.environ["EMBEDDING_AZURE_DEPLOYMENT"],
        api_version=os.environ["EMBEDDING_AZURE_API_VERSION"],
        openai_api_type="azure",
        model=os.environ["EMBEDDING_AZURE_MODEL"],
        api_key=os.environ["EMBEDDING_AZURE_API_KEY"],
    )

    vector_store = LanceDB(uri=uri, embedding=embeddings, table_name=table_name)
    return vector_store
