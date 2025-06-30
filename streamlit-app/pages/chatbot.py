import streamlit as st
from langchain_core.messages import HumanMessage
from supervisor.supervisor_graph import build_graph  # Your LangGraph app

# Build the agent graph
app = build_graph()

# Initialize graph state
if "graph_state" not in st.session_state:
    st.session_state.graph_state = {
        "messages": [],
        "input": "",
        "last_agent": "start",
    }

st.title("🏥 Healthcare Data Assistant")

# Take user input
if user_input := st.chat_input("Ask a question about healthcare operations…"):
    st.session_state.graph_state["messages"] = [HumanMessage(content=user_input)]
    st.session_state.graph_state["input"] = user_input
    st.session_state.graph_state["last_agent"] = "start"

    with st.spinner("Processing with IntentFinder..."):
        st.session_state.graph_state = app.invoke(st.session_state.graph_state)
        response = st.session_state.graph_state["output"]
        st.chat_message("assistant").write(response)

# Buttons to progress through agent chain
if st.button("➡️ Agent 2: ContextFinder"):
    st.session_state.graph_state["messages"].append(HumanMessage(content="proceed"))
    st.session_state.graph_state["last_agent"] = "agent_1"
    st.session_state.graph_state = app.invoke(st.session_state.graph_state)
    st.chat_message("assistant").write(st.session_state.graph_state["output"])

if st.button("➡️ Agent 3: SQLQueryGenerator"):
    st.session_state.graph_state["messages"].append(HumanMessage(content="proceed"))
    st.session_state.graph_state["last_agent"] = "agent_2"
    st.session_state.graph_state = app.invoke(st.session_state.graph_state)
    st.chat_message("assistant").write(st.session_state.graph_state["output"])

if st.button("➡️ Agent 4: SQLQueryExecutor"):
    st.session_state.graph_state["messages"].append(HumanMessage(content="proceed"))
    st.session_state.graph_state["last_agent"] = "agent_3"
    st.session_state.graph_state = app.invoke(st.session_state.graph_state)
    st.chat_message("assistant").write(st.session_state.graph_state["output"])

if st.button("➡️ Agent 5: PlotGenerator"):
    st.session_state.graph_state["messages"].append(HumanMessage(content="proceed"))
    st.session_state.graph_state["last_agent"] = "agent_4"
    st.session_state.graph_state = app.invoke(st.session_state.graph_state)
    st.chat_message("assistant").write(st.session_state.graph_state["output"])

# Reset session
if st.button("🔄 Reset Session"):
    st.session_state.graph_state = {
        "messages": [],
        "input": "",
        "last_agent": "start",
    }
    st.rerun()
