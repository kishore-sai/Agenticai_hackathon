import functools
from typing import List

from langchain.agents.react.agent import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate

from datachat.agents.context_finder.tools import context_finder_tool
from datachat.agents.helpers import get_llm, invoke_agent

AGENT_NAME = "contextFinder"
llm = get_llm()

context_finder_agent_prompt = PromptTemplate(
    template="""
        You are a data analyst who can generate a list of documents for the user's question by utilizing the available tools.
        You have access to the following tools:
        {tools}


        Thought: you should always think about what to do. Make sure to include the previous tool results

        Action: the action to take, should be one of [{tool_names}]. Do not give me any descriptive statements, explanation or justification for picking any tool. Just give the tool name as the action.

        Action Input: The input to the action.Start with the input question and enhance it by making it more descriptive and structured, without changing the objective. Identify key entities, attributes, and relationships ,For  financial metrics Search through the table descriptions to find tables that mention concepts like `sales`, `revenue`, `payment`, `transactions`, `time`, `date`,etc. relevant to the query. Reformulate the question in a way that facilitates the retrieval of necessary table metadata for generating an accurate SQL query.

        Critique: The critique to generate best result

        Observation: the result of the action


        ... (this Thought/Action/Action Input/Observation may repeat N times and you can call the same tool multiple times if you are not satisfied with the output.)

        When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:


        Thought: I now know the final answer

        Final Answer: the final answer to the original input question


        Begin!

        Input question:{input}
        Thought: {agent_scratchpad}

        Check your output and make sure it conforms! Do not output an action and a final answer at the same time.
        Note: Do not return as json markdown
        """
)


def context_finder_node(state, agent, tools, name):
    print("INSIDE AGENT TO NODE OF", name)
    result, observation = invoke_agent(agent, state, tools)
    print("AGENT OBSERVATION:", observation)
    # If our observation is as we expect, the agent has finished its work. Trigger an AgentFinish object
    if isinstance(observation, list):
        return {
            "docs": observation,
            # HumanMessage with the agent name is required for the supervisor to recognize the agent output
            # Refer: https://vijaykumarkartha.medium.com/hierarchical-ai-agents-create-a-supervisor-ai-agent-using-langchain-315abbbd4133
            "messages": [
                HumanMessage(
                    content="Document context generated successfully, added in state variable and can be proceeded to SQL Generation",
                    name=name,
                )
            ],
        }
    
    print("Agent result:", result)
    # If the observation didn't meet our expectation, pass the outcome as is for now
    return {"intermediate_steps": result}


agent_tools = [context_finder_tool]

context_finder_agent = create_react_agent(
    llm, tools=agent_tools, prompt=context_finder_agent_prompt
)
context_finder__agent_node = functools.partial(
    context_finder_node, tools=agent_tools, agent=context_finder_agent, name=AGENT_NAME
)
