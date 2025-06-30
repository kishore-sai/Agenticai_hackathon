import ast
import functools
import json

from langchain.agents import (
    AgentExecutor,
    create_openai_tools_agent,
    create_react_agent,
    create_tool_calling_agent,
    tool_calling_agent,
)
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import ToolNode

from datachat.agents.helpers import get_llm, invoke_agent
from datachat.agents.validate_query.tools import validate_query

AGENT_NAME = "sqlValidator"


def validator_tool_node(state, name):
    print("INSIDE AGENT", AGENT_NAME)

    result, msg = validate_query(state["sql_query"], state["docs"])
    print("Agent Output:", result, msg)
    return {"messages": [HumanMessage(content=msg, name=name)]}


sql_validator_node = functools.partial(validator_tool_node, name=AGENT_NAME)
