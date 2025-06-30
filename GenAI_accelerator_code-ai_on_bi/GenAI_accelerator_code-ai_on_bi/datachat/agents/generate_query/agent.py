import ast
import functools
import json

from langchain.agents.react.agent import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.utils.json import parse_and_check_json_markdown

from datachat.agents.generate_query.tools import (
    QueryAndExplanation,
    generate_query_for_question,
)
from datachat.agents.helpers import get_llm, invoke_agent

AGENT_NAME = "sqlGenerator"
llm = get_llm()

sql_agent_prompt = PromptTemplate(
    template="""
        You are a data analyst who can generate a valid SQL query in SQL Server dialect(Tsql) for the user's question by utilizing the available tools. But you don't have permission to execute that generated query.
        You have access to the following tools:
        {tools}


        Thought: you should always think about what to do. Make sure to include the previous tool results

        Action: the action to take, should be one of [{tool_names}]. Do not give me any descriptive statements, explanation or justification for picking any tool. Just give the tool name as the action.

        Action Input: The input to the action. Never truncate the action input to the given tool. Pass the entire available arguments to the tool. Always a python loadable json. Don't truncate the context documents even if it is large, pass the entire value

        Observation: the result of the action


        ... (this Thought/Action/Action Input/Observation may repeat N times and you can call the same tool multiple times if you are not satisfied with the output.)
        
        Begin!

        Input question:{input}
        Context Documents: {docs}
        Thought: {agent_scratchpad}

        Check your output and make sure it conforms! Do not output an action and a final answer at the same time. Need to return the action and action input mandatorily without changing or truncating it.
        Note: Do not return as json markdown 
        """
)


def action_input_parser(action_input: str) -> dict:
    # Strip empty spaces and any trailing quotes
    cleaned = action_input.replace("python", "").strip()
    return parse_and_check_json_markdown(
        cleaned, expected_keys=["question", "docs"]
    )


def generator_agent_node(state, agent, tools, name):
    print("INSIDE AGENT TO NODE OF", name)
    result, observation = invoke_agent(
        agent, state, tools, input_preprocessor=action_input_parser
    )
    print("AGENT OBSERVATION:", observation)
    # If our observation is as we expect, the agent has finished its work. Trigger an AgentFinish object
    if isinstance(observation, QueryAndExplanation):
        return {
            "sql_query": observation.query,
            "query_explanation": observation.explanation,
            # HumanMessage with the agent name is required for the supervisor to recognize the agent output
            # Refer: https://vijaykumarkartha.medium.com/hierarchical-ai-agents-create-a-supervisor-ai-agent-using-langchain-315abbbd4133
            "messages": [HumanMessage(content=observation.query, name=name)],
        }

    print("Agent result:", result)
    # If the observation didn't meet our expectation, pass the outcome as is for now
    return {"intermediate_steps": result}


agent_tools = [generate_query_for_question]

sql_generator_agent = create_react_agent(
    llm, tools=agent_tools, prompt=sql_agent_prompt
)
sql_generator_node = functools.partial(
    generator_agent_node, tools=agent_tools, agent=sql_generator_agent, name=AGENT_NAME
)
