import functools
import re
import pymssql
import snowflake
from langchain.agents.react.agent import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate

from datachat.agents.execute_query.tools import DataFrameInfo, query_db
from datachat.agents.helpers import get_llm, invoke_agent

AGENT_NAME = "sqlExecutor"
llm = get_llm()

sql_agent_prompt = PromptTemplate(
    template="""
        You are a data analyst. Your task is to execute a SQL Query statement from a QueryandExplanation object and return the result of the query as a pandas dataframe
        You have access to the following tools:
        {tools}


        Thought: you should always think about what to do. Make sure to include the previous tool results

        Action: the action to take, should be one of [{tool_names}]. Do not give me any descriptive statements, explanation or justification for picking any tool. Just give the tool name as the action.

        Action Input: The query to be executed. Start with the sql query

        Observation: the result of the action


        ... (this Thought/Action/Action Input/Observation may repeat N times and you can call the same tool multiple times if you are not satisfied with the output.)

        When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:


        Thought: I now know the final answer

        Final Answer: the finalized sql query to the original input question.


        Begin!

        Input: {sql_query}
        Thought: {agent_scratchpad}

        Check your output and make sure it conforms! Do not output an action and a final answer at the same time.
        """
)
# The sql_query in the prompt template is populated from the agent state variable


def action_input_parser(action_input: str) -> str:
    # Sometimes the input is formatted as `query = <actual query>`
    pattern = re.search(
        pattern=r"^\s*\"?query\s*=\s*(.+)$", string=action_input.strip()
    )

    if pattern:
        print("Found query= prefix in the action input. Extracting only the query")
        return pattern.group(1)

    return action_input


def executor_agent_node(state, agent, tools, name):
    print("INSIDE AGENT TO NODE OF", name)
    result, observation = invoke_agent(
        agent, state, tools, input_preprocessor=action_input_parser
    )
    print("AGENT OBSERVATION:", observation)
    # If our observation is as we expect, the agent has finished its work. Trigger an AgentFinish object
    if isinstance(observation, DataFrameInfo):
        return {
            "df_path": observation.path,
            "messages": [HumanMessage(content=observation.info, name=name)],
            "columns": observation.columns,
        }

    if isinstance(observation, pymssql.ProgrammingError):
        return {
            "messages": [
                HumanMessage(
                    content=" ".join(
                        [
                            "The SQL query output from the sqlGenerator agent is not valid. ",
                            f"The error in the query is {observation}."
                            "The SQL query needs to be regeneated with this error fixed",
                        ]
                    ),
                    name=name,
                )
            ]
        }

    print("Agent result:", result)
    # If the observation didn't meet our expectation, pass the outcome as is for now
    return {"intermediate_steps": result}


agent_tools = [query_db]

sql_executor_agent = create_react_agent(llm, tools=agent_tools, prompt=sql_agent_prompt)
sql_executor_node = functools.partial(
    executor_agent_node, tools=agent_tools, agent=sql_executor_agent, name=AGENT_NAME
)
