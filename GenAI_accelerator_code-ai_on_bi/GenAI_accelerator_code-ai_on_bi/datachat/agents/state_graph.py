import os
from typing import Literal

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.errors import GraphRecursionError
from numpy import dtype
from datachat.agents.context_finder.agent import context_finder__agent_node
from datachat.agents.execute_query.agent import sql_executor_node
from datachat.agents.generate_plot.agent import visualization_generator_node
from datachat.agents.generate_query.agent import sql_generator_node
from datachat.agents.helpers import AgentState
from datachat.agents.supervisor import members, supervisor_agent
from datachat.agents.validate_query.agent import sql_validator_node
from datachat.utils.vectordb import init_vector_store

LANCE_URI = os.environ["LANCE_URI"]
TABLE_NAME = os.environ["VECTORDB_TABLE_NAME"]

vector_store = init_vector_store(LANCE_URI, TABLE_NAME)


def route_supervisor(state) -> Literal["call_tool", "__end__", "continue"]:
    print("ROUTING SUPERVISOR BASED ON STATE", state)
    if "fig_path" in state:
        return "FINISH"
    elif "df_path" in state and (
         len(state["columns"]) == 0
    ):
        return "FINISH"
    elif "df_path" in state and (
        all(v == dtype('O') for v in state["columns"].values())
    ):
        return "FINISH"
    return state["next"]


workflow = StateGraph(AgentState)
workflow.add_node("contextFinder", context_finder__agent_node)
workflow.add_node("sqlGenerator", sql_generator_node)
workflow.add_node("sqlValidator", sql_validator_node)
workflow.add_node("sqlExecutor", sql_executor_node)
workflow.add_node("visualizationGenerator", visualization_generator_node)
workflow.add_node("supervisor", supervisor_agent)

for member in members:
    workflow.add_edge(member, "supervisor")


conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
print("CONDITIONAL_MAP", conditional_map)
workflow.add_conditional_edges("supervisor", route_supervisor, conditional_map)
workflow.add_edge(START, "supervisor")

graph = workflow.compile()

# graph.get_graph().draw_png("image.png")


def generate_output(user_question):
    try:
        result_var = graph.invoke(
            {
                "messages": [HumanMessage(content=user_question)],
                "input": user_question,
            }
        )

        # print("Out")
        print("Answer====>", result_var)
        # print(final_result)
        return result_var
    except GraphRecursionError as e:
        return "Give some extra context to the question"
