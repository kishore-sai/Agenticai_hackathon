import base64
import json
import logging
import os
from datetime import datetime
import os
import lancedb
import pyarrow as pa
from io import BytesIO
import sqlparse

import pandas as pd
import streamlit as st
from langgraph.errors import GraphRecursionError
from pandas.errors import DatabaseError
from PIL import Image
from langchain_core.messages import HumanMessage
from langchain.embeddings import AzureOpenAIEmbeddings
import plotly.express as px

from datachat.agents.state_graph import generate_output, graph
from datachat.utils.db import add_to_chat_history, get_chat_history
from datachat.utils.vectordb import init_vector_store
from datachat.agents.helpers import get_llm

import plotly.graph_objs as go
import plotly.io as pio


def init_state():
    def set_default_if_not_present(key, default_val=""):
        if key not in st.session_state:
            st.session_state[key] = default_val
    set_default_if_not_present("state_docs", None)
    set_default_if_not_present("graph_state", dict())
    set_default_if_not_present("context_docs", dict())
    set_default_if_not_present("docs", dict())
    set_default_if_not_present("response", None)
    set_default_if_not_present("human", None)
    set_default_if_not_present("user_input", "")
    set_default_if_not_present("awaiting_human", False)
    set_default_if_not_present("result", None)
    set_default_if_not_present("add_docs", dict())
    set_default_if_not_present("sql_query", dict())
    set_default_if_not_present("chat_history", [])
    set_default_if_not_present("explanation", "")
    set_default_if_not_present("context_interupt", None)
    set_default_if_not_present("selected_function", "")
    set_default_if_not_present("plot", dict())
    set_default_if_not_present("similar_inputs", pd.DataFrame(columns=["score", "user_input", "sql_query"]))
init_state()


# === CONFIG ===
VECTOR_DIM = 1536
DB_PATH = "lance_db"
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



llm = get_llm()
start_time = datetime.now()
# Set full page width
st.set_page_config("Chat2Data", layout="wide")

# Page title
st.title("💬 Chat2Data")
# chat_history_container = st.container()

logger = logging.getLogger(__file__)

LANCE_URI = os.environ["LANCE_URI"]
TABLE_NAME = "suits"

vector_store = init_vector_store(LANCE_URI, TABLE_NAME)
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

def add_to_db(user_input, sql_query):
    embedding = generate_embeddings(user_input)
    table.add([{
        "user_input": user_input,
        "sql_query": sql_query,
        "embedding": embedding
    }])
    print(f"✅ Added: {user_input}")

def decode_image(base64_string):
    buf = BytesIO(base64.b64decode(base64_string))
    img = Image.open(buf)
    return img

def decode_plotly(retrieved_graph):
    if retrieved_graph is None:
        return None
    # Deserialize the binary back to JSON
    retrieved_graph_json = json.loads(retrieved_graph.decode('utf-8')) #.graph_json

    # Create a Plotly figure from the JSON data
    retrieved_fig = pio.from_json(retrieved_graph_json)

    return retrieved_fig

def calculate_response(user_question):
    response = generate_output(user_question)
    if type(response) is str:
        return response
    logger.info(f"Generated query for user question is {response['sql_query']}")
    try:
        logger.info(response["df_path"])
        op_df = pd.read_pickle(response["df_path"])
    except DatabaseError as ex:
        response["query_explanation"] = "Couldn't connect to the database"
        fig = None
        return response, fig
    fig = None
    Error_response = ""
    if len(op_df) > 0:
        if op_df.shape[1] > 1:
            fig = response["fig_path"]
        df_response = op_df
    else:
        df_response = "Query produced no results"
    chart_exp = llm.invoke(
        f"""For the given sql query explanation {response['query_explanation']} and the dataframe columns {response['columns']} generated using the sql query and the plot type {response['plot_type']} used for visualizing the dataframe, Give a one line chart description"
    """).content
    return response["sql_query"], response["query_explanation"], fig, df_response,chart_exp

CSS = """
.stChatMessage:has([data-testid="stChatMessageAvatarUser"]) {
    display: flex;
    flex-direction: row-reverse;
    align-itmes: end;
}

[data-testid="stChatMessageAvatarUser"] + [data-testid="stChatMessageContent"] {
    text-align: right;
}
"""
st.html(f"<style>{CSS}</style>")

def reset_session_state():
    keys = list(st.session_state.keys())
    for key in keys:
        if key not in("user_input", "chat_history"):
            del st.session_state[key]

def display_chat_history():
    # messages = get_chat_history()
    # for chat in reversed(messages):
    #     if chat != messages[-1] and chat['user_question']:
    #         with st.chat_message(name="user"):
    #             st.write(chat['user_question'])
    #     elif len(st.session_state.chat_history)==0:
    #         with st.chat_message(name="user"):
    #             st.write(chat['user_question'])
    #     with st.chat_message(name="assistant"):
    #         if isinstance(decode_plotly(chat['figure']), go.Figure):
    #             st.plotly_chart(decode_plotly(chat['figure']))
    #         if chat['query']:
    #             with st.expander(chat['figexp']):
    #                 formatted_query = sqlparse.format(chat['query'], reindent=True, keyword_case='upper')
    #                 st.markdown(
    #                     f"<pre><code class='language-sql'>{formatted_query}</code></pre>",
    #                     unsafe_allow_html=True
    #                 )
    #                 st.markdown(
    #                     f"<pre style='white-space: pre-wrap; word-break: break-word;'>{chat['explanation']}</pre>",
    #                     unsafe_allow_html=True
    #                 )
    #         if chat == messages[-1] and len(st.session_state.chat_history)>0:
    #             data = st.session_state.chat_history[-1]
    #             st.dataframe(data['op_df'])
    if len(st.session_state.chat_history)>0:
        for i, chat in enumerate(st.session_state.chat_history):
            with st.chat_message(name="user"):
                st.write(chat['user'])
            with st.chat_message(name="assistant"):
                if chat['plot_fig']:
                    if i == len(st.session_state.chat_history) - 1:
                        if bool(st.session_state.plot):
                            data = st.session_state.chat_history
                            df = data['op_df']

                            # Handle filter columns
                            filter_columns = st.session_state.plot.get('filter_col', [])
                            if isinstance(filter_columns, str):
                                filter_columns = [filter_columns]

                            filter_values = {}
                            for col in filter_columns:
                                options = df[col].unique().tolist()
                                selected = st.multiselect(f"Select {col}(s):", options=options, default=options)
                                filter_values[col] = selected

                            filtered_df = df.copy()
                            for col, selected_values in filter_values.items():
                                if selected_values:
                                    filtered_df = filtered_df[filtered_df[col].isin(selected_values)]
                            if op_df.shape == (1, 1):
                                # Extract value and label
                                value = op_df.iloc[0, 0]
                                label = op_df.columns[0]

                                # Create an indicator plot
                                fig = go.Figure(go.Indicator(
                                    mode="number",
                                    value=value,
                                    title={"text": label}
                                ))
                            # Build new figure using Plotly Express
                            elif filtered_df.shape[0] < 2 and st.session_state.plot['plot_type'] in ['pie', 'heatmap']:
                                raise ValueError("Not enough data points to plot a meaningful pie or heatmap.")
                            elif st.session_state.plot['plot_type'] == 'scatter':
                                fig = px.scatter(filtered_df,  x=st.session_state.plot['x'], y=st.session_state.plot['y'], color=st.session_state.plot['color'])
                            elif st.session_state.plot['plot_type'] == 'line':
                                fig = px.line(filtered_df,  x=st.session_state.plot['x'], y=st.session_state.plot['y'], color= st.session_state.plot['color'])
                            elif st.session_state.plot['plot_type'] == 'bar':
                                fig = px.bar(filtered_df,  x=st.session_state.plot['x'],y=st.session_state.plot['y'], color=st.session_state.plot['color'],  barmode="group")
                            elif st.session_state.plot['plot_type'] == 'pie':
                                fig = px.pie(filtered_df, names=st.session_state.plot['x'], values=st.session_state.plot['y'])
                            elif st.session_state.plot['plot_type'] == 'heatmap':
                                fig = px.density_heatmap(filtered_df,  x=st.session_state.plot['x'], y=st.session_state.plot['y'])

                            st.plotly_chart(fig)
                        else:
                            st.plotly_chart(chat['plot_fig'])
                    else:
                        st.plotly_chart(chat['plot_fig'])
                if chat['query']:
                    with st.expander(chat['chart_exp']):
                        formatted_query = sqlparse.format(chat['query'], reindent=True, keyword_case='upper')
                        st.markdown(
                            f"<pre><code class='language-sql'>{formatted_query}</code></pre>",
                            unsafe_allow_html=True
                        )
                        st.markdown(
                            f"<pre style='white-space: pre-wrap; word-break: break-word;'>{chat['explanation']}</pre>",
                            unsafe_allow_html=True
                        )
                    st.dataframe(chat['op_df'])

# List of function-level LanceDB table names
function_tables = {
    "Sales & Revenue Performance": "sales_revenue_performance",
    "Customer Insights": "customer_insights",
    "Product & Category Analytics": "product_category_analytics",
    "Others": "table2"
}

# Dropdown on the first page
with st.sidebar:
    selected_function = st.selectbox(
        "Select a business function to continue:",
        options=["-- Select --"] + list(function_tables.keys())
    )
# Show selected value
#st.write(f"🔍 You selected: `{selected_function}`")
#st.write(f"LanceDB Table Name: `{function_tables[selected_function]}`")

display_chat_history()

if selected_function != "-- Select --" :
    if user_input := st.chat_input("Type your message..."):
        reset_session_state()
        init_state()
        st.session_state.selected_function=function_tables[selected_function]
        st.session_state.user_input=user_input
        prompt=f"""You are a data assistant that verifies whether a business question fits a specific analytical function.

        Given the business question:
        "{user_input}"

        And the selected function:
        "{selected_function}"

        Determine if the question is related to the purpose of the given function. Respond with:
        - `True` if the question is likely relevant to the function.
        - `False` if the question is not clearly related to the function.

        Only respond with `True` or `False`. Do not include any explanation."""
        
        business_func_validation=llm.invoke(prompt).content
        print(business_func_validation)
        if st.session_state.user_input!="":
            with st.chat_message(name="user"):
                st.write(st.session_state.user_input)
        if business_func_validation=="True" or st.session_state.selected_function=="table2":
            with st.spinner("Thinking..."):
                response = generate_output(st.session_state.user_input)
                st.session_state.response=response
                st.session_state.graph_state = response
                st.session_state.result = response
                # st.write(st.session_state.result)
        else:
            st.info("The question asked is not related to selected business functionality")

        # if st.session_state.user_input!="":
        #     with st.chat_message(name="user"):
        #         st.write(st.session_state.user_input)
    # if st.session_state.user_input!="":
    #     with st.chat_message(name="user"):
    #         st.write(st.session_state.user_input)


    # Human input section
    if st.session_state.awaiting_human==True:
        if st.session_state.context_docs:
            if not isinstance(st.session_state.context_docs, dict):
                st.session_state.context_docs = json.loads(st.session_state.context_docs)
            st.info(st.session_state.context_docs["explanation"])
            columns_to_select=["table","column","description"]
            #st.write(type(st.session_state.docs))
            #st.write(st.session_state.docs)
            st.dataframe([{key: item[key] for key in columns_to_select if key in item} for item in st.session_state.docs])
            extra_context=st.text_input("give additional context")  
            if st.button("Submit Review"):
                with st.spinner("Thinking..."):
                    st.session_state.graph_state["input"] = st.session_state.graph_state["input"]+extra_context
                    result = graph.invoke({
                                **st.session_state.graph_state,
                            })
                    st.session_state.graph_state = result
                    st.session_state.awaiting_human = True
                    st.session_state.result = result
                    st.rerun()
        elif st.session_state.context_interupt!=None:
            formatted_query = sqlparse.format(st.session_state.sql_query, reindent=True, keyword_case='upper')
            st.markdown(
                f"<pre><code class='language-sql'>{formatted_query}</code></pre>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<pre style='white-space: pre-wrap; word-break: break-word;'>{st.session_state.explanation}</pre>",
                unsafe_allow_html=True
            )
            st.session_state.extra_feedback = ""
            approved = st.radio("Approve this query?", ("yes", "no"), key="approval_choice")
            extra = st.text_area("If not approved, add feedback:", "", key="extra_feedback")

            if st.button("Submit Review"):
                with st.spinner("Thinking..."):
                    approved_flag = True if approved == "yes" else False
                    print("???????????????????????????????????")
                    #st.write(st.session_state.graph_state)
                    if approved=="yes":
                        st.session_state.awaiting_human = False
                        st.session_state.human=True
                        st.session_state.graph_state["next"] = "sqlValidator"
                        st.session_state.graph_state["sql_query"] = st.session_state.sql_query
                        st.session_state.graph_state["query_explanation"]=st.session_state.explanation
                        new_sql_message = HumanMessage(
                            content=st.session_state.graph_state["sql_query"],
                            name="sqlGenerator"
                        )
                        st.session_state.graph_state["messages"].append(new_sql_message)
                    else:
                        st.session_state.graph_state["input"] = st.session_state.graph_state["input"]+extra
                    try:
                        result = graph.invoke({
                            **st.session_state.graph_state,
                        })
                        st.session_state.result = result
                        st.rerun()
                    except GraphRecursionError as e:
                        st.session_state.result = None
        else:
            try:
                if 'sql_query' in st.session_state.result.keys():
                    logger.info(f"Generated query for user question is {st.session_state.result['sql_query']}")
                    try:
                        logger.info(st.session_state.result["df_path"])
                        op_df = pd.read_pickle(st.session_state.result["df_path"])
                    except DatabaseError as ex:
                        st.session_state.result["query_explanation"] = "Couldn't connect to the database"
                        fig = None
                    fig = None
                    Error_response = ""
                    if len(op_df) > 0:
                        if op_df.shape[1] > 1:
                            fig = st.session_state.result["fig_path"]
                        df_response = op_df
                    else:
                        df_response = "Query produced no results"
                    query=st.session_state.result["sql_query"]
                    explanation=st.session_state.result["query_explanation"]
                    user_input=st.session_state.user_input
                    if 'plot_type' in st.session_state.result.keys():
                        chart_exp = llm.invoke(
                        f"""For the given sql query explanation {st.session_state.result['query_explanation']} and the sql query {st.session_state.result['sql_query']} generated using the sql query and the plot type {st.session_state.result['plot_type']} used for visualizing the dataframe, Give a one line chart description for the visual"
                        """).content
                    else:
                        chart_exp ="Chart is not generated as the dataframe having one column or null data"
                    add_to_chat_history(
                        messages=[(query, explanation, user_input,chart_exp)], figure=fig
                    )
                    
                    st.session_state.chat_history.append({
                        'user': user_input,
                        'query': query,
                        'explanation': explanation,
                        'plot_fig': fig,
                        'chart_exp': chart_exp,
                        'op_df' : op_df
                    })
                    st.session_state.result =None
                    add_to_db(user_input,query)
                    st.rerun()
                else:
                    st.write(st.session_state.result['explanation'])
            except ValueError as e:
                st.write(e)
                st.write(st.session_state.result)
                #add_to_chat_history(
                #    messages=[("Error", final_output, user_input)]
                #)

    if st.session_state.result !=None and st.session_state.awaiting_human==False:
        try:
            if 'sql_query' in st.session_state.result.keys():
                logger.info(f"Generated query for user question is {st.session_state.result['sql_query']}")
                try:
                    logger.info(st.session_state.result["df_path"])
                    op_df = pd.read_pickle(st.session_state.result["df_path"])
                except DatabaseError as ex:
                    st.session_state.result["query_explanation"] = "Couldn't connect to the database"
                    fig = None
                fig = None
                Error_response = ""
                if len(op_df) > 0:
                    if op_df.shape[1] > 1:
                        fig = st.session_state.result["fig_path"]
                    df_response = op_df
                else:
                    df_response = "Query produced no results"
                query=st.session_state.result["sql_query"]
                explanation=st.session_state.result["query_explanation"]
                user_input=st.session_state.user_input
                if 'plot_type' in st.session_state.result.keys():
                    chart_exp = llm.invoke(
                    f"""For the given sql query explanation {st.session_state.result['query_explanation']} and the sql query {st.session_state.result['sql_query']} generated using the sql query and the plot type {st.session_state.result['plot_type']} used for visualizing the dataframe, Give a one line chart description for the visual"
                    """).content
                else:
                    chart_exp ="Chart is not generated as the dataframe having one column or null data"
                add_to_chat_history(
                    messages=[(query, explanation, user_input,chart_exp)], figure=fig
                )
                
                st.session_state.chat_history.append({
                    'user': user_input,
                    'query': query,
                    'explanation': explanation,
                    'plot_fig': fig,
                    'chart_exp': chart_exp,
                    'op_df' : op_df
                })
                st.session_state.result =None
                print("hii")
                add_to_db(user_input,query)
                st.rerun()
            else:
                st.write(st.session_state.result['explanation'])
        except ValueError as e:
            st.write(e)
            st.write(st.session_state.result)
            #add_to_chat_history(
            #    messages=[("Error", final_output, user_input)]
            #)
        
