import base64
import json
import logging
import os
from datetime import datetime
from io import BytesIO

import duckdb
import lancedb
import pandas as pd
import streamlit as st
from pandas.errors import DatabaseError
from PIL import Image
from streamlit_chat import message

from datachat.agents.state_graph import generate_output
from datachat.utils.db import add_to_chat_history, get_chat_history
from datachat.utils.vectordb import init_vector_store

st.set_page_config("Chat2Data", layout="wide")

logger = logging.getLogger(__file__)

LANCE_URI = os.environ["LANCE_URI"]
TABLE_NAME = os.environ["VECTORDB_TABLE_NAME"]
chat_history_container = st.container()
user_input_container = st.container()

vector_store = init_vector_store(LANCE_URI, TABLE_NAME)


def decode_image(base64_string):
    buf = BytesIO(base64.b64decode(base64_string))
    img = Image.open(buf)
    return img


def as_chat_message(query, explanation):
    return f"""{explanation}
        ```
        {query}
        ```
        """


def calculate_response():
    response = generate_output(user_question)
    logger.info(f"Generated query for user question is {response['sql_query']}")
    chat_history_container.code(response["sql_query"])
    chat_history_container.text(response["query_explanation"])
    try:
        logger.info(response["df_path"])
        op_df = pd.read_pickle(response["df_path"])
    except DatabaseError as ex:
        response["query_explanation"] = "Couldn't connect to the database"
        fig = None
        chat_history_container.text(response["query_explanation"])
        return response, fig
    fig = None
    if len(op_df) > 0:
        if op_df.shape[1] > 1:
            fig = Image.open(response["fig_path"])
            chat_history_container.image(fig)
        chat_history_container.dataframe(op_df)
    else:
        chat_history_container.text("Query produced no results")

    return response["sql_query"], response["query_explanation"], fig


def show_chat_history():
    logger.info("Fetching chat history")
    messages = get_chat_history()
    with chat_history_container:
        for msg in messages:
            logger.info(f"Got {msg} from chat history")
            if msg.figure:
                chat_history_container.image(image=decode_image(msg.figure))
            message(message=msg.content, is_user=msg.is_user, key=msg.id)


show_chat_history()
user_question = user_input_container.text_input("Ask a question", key="user_input")

if user_question:
    logger.info("Adding user question to chat history")
    add_to_chat_history(messages=[(user_question, True)])
    query, explanation, plot_fig = calculate_response()
    add_to_chat_history(
        messages=[(as_chat_message(query, explanation), False)], figure=plot_fig
    )
    logger.info("Added system response to chat history")
