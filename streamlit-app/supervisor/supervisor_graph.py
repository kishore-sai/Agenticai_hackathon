from langgraph.graph import StateGraph, START, END
from agents.intent import agent_node as agent_1_node
from agents.context_finder import agent_node as agent_2_node
from agents.sql_generator import agent_node as agent_3_node
from agents.sql_executor import agent_node as agent_4_node
from agents.plot_generator import agent_node as agent_5_node
from utils.routing import router
from langgraph.graph import StateGraph
from typing import TypedDict, List
from langchain_core.messages import BaseMessage


# Define state schema
class AgentState(TypedDict):
    messages: List[BaseMessage]
    sessionId: str
    last_agent: str


def build_graph():
    graph = StateGraph(state_schema=AgentState)

    graph.add_node("agent_1", agent_1_node)
    graph.add_node("agent_2", agent_2_node)
    graph.add_node("agent_3", agent_3_node)
    graph.add_node("agent_4", agent_4_node)
    graph.add_node("agent_5", agent_5_node)

    graph.set_entry_point("agent_1")

    graph.add_conditional_edges("agent_1", router)
    graph.add_conditional_edges("agent_2", router)
    graph.add_conditional_edges("agent_3", router)
    graph.add_conditional_edges("agent_4", router)

    graph.set_finish_point("agent_5")

    return graph.compile()
