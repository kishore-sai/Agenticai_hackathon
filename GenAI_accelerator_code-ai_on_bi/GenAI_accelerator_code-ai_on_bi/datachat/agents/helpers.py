import operator
import os
from typing import Annotated, Literal, Sequence, TypedDict, Union

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import AzureChatOpenAI


def get_llm():
    llm = AzureChatOpenAI(
        azure_endpoint=os.environ["LLM_AZURE_ENDPOINT"],
        azure_deployment=os.environ["LLM_AZURE_DEPLOYMENT"],
        api_version=os.environ["LLM_AZURE_API_VERSION"],
        openai_api_type="azure",
        model=os.environ["LLM_AZURE_MODEL"],
        verbose=True,
        temperature=0
    )
    return llm


def invoke_agent(agent, state, tools, input_preprocessor=lambda x: x):
    print(f"INVOKING AGENT {agent} WITH {state}")
    result = agent.invoke(state)
    tools_dict = {t.name: t for t in tools}
    pre_processed_input = input_preprocessor(result.tool_input)
    observation = tools_dict[result.tool].invoke(pre_processed_input)
    return result, observation


def agent_to_node(state, agent, tools, name):
    print("INSIDE AGENT TO NODE OF", name)
    # global final_result
    result = agent.invoke(state)
    observation = {t.name: t for t in tools}[result.tool.strip(".")].invoke(
        result.tool_input
    )
    print("AGENT OBSERVATION:", observation)
    return {"agent_outcome": result}


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[
        list[tuple[AgentAction | AgentFinish, str]], operator.add
    ]
    # Agent to act next
    next: str
    # User input
    input: str
    # Context Documents generated for the user question
    docs: list
    # SQL query generated for the user question
    # Agent Variable for sqlgenerator output
    sql_query: str
    # Explanation for the sql query
    query_explanation: str
    # Dataframe path where the result set is stored
    df_path: str
    # Dataframe Columns with datatype
    columns: dict
    # Plot Type
    plot_type: str
    fig_path: str
    # NEW FIELDS for humanAgent
    human_review_result: Literal["approved", "needs_revision"]
    extra_context: str