import ast
import functools
import json

from langchain.agents.react.agent import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
import plotly.graph_objs as go

from datachat.agents.generate_plot.tools import create_chart
from datachat.agents.helpers import get_llm, invoke_agent

AGENT_NAME = "visualizationGenerator"
llm = get_llm()

visualization_agent_prompt = PromptTemplate(
    template="""
        You are a Data Vizualization expert. For the following question, choose the type of visualization and the plot dimensions for the columns from the column names by utilizing the available tools.
        You have access to the following tools:
        {tools}
        The order of execution of tools {tool_names}

        Thought: you should always think about what to do. Make sure to include the previous tool results

        Action: the action to take, should be one of [{tool_names}]. Do not give me any descriptive statements, explanation or justification for picking any tool. Just give the tool name as the action.

        Action Input: the input to the action. Always a python loadable json.

        Observation: the result of the action


        ... (this Thought/Action/Action Input/Observation may repeat N times and you can call the same tool multiple times if you are not satisfied with the output.)

        When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:


        Thought: I now know the final answer

        Final Answer: the final answer to the original input question


        Begin!

        Input question: {input}
        Columns: {columns}
        Dataframe Path: {df_path}
        Thought: {agent_scratchpad}

        Check your output and make sure it conforms! Do not output an action and a final answer at the same time.
        """
)


def action_input_parser(action_input: str) -> dict:
    # Strip empty spaces and any trailing quotes
    return ast.literal_eval(
        action_input.replace("Output:", "").replace("python", "").strip("`").strip()
    )


def visualization_agent_node(state, agent, tools, name):
    print("INSIDE AGENT TO NODE OF", name)
    # HACK: Add a proper parsing logic for inputs
    result, observation = invoke_agent(
        agent,
        state,
        tools,
        input_preprocessor=action_input_parser,
    )
    print("AGENT OBSERVATION:", observation)
    if isinstance(observation, str):
        return {
            "fig_path": observation,
            "messages": [
                HumanMessage(
                    content=f"The chart is generated and stored in the following path {observation}.",
                    name=name,
                )
            ],
        }
    elif isinstance(observation, go.Figure):
        return {
            "fig_path": observation,
            "messages": [
                HumanMessage(
                    content=f"Above is the Plolty Chart",
                    name=name,
                )
            ],
            "plot_type": observation.data[0].type
        }
    print("Agent result:", result)
    # If the observation didn't meet our expectation, pass the outcome as is for now
    return {"messages": result}


agent_tools = [create_chart]

visualization_generator_agent = create_react_agent(
    llm, tools=agent_tools, prompt=visualization_agent_prompt
)
visualization_generator_node = functools.partial(
    visualization_agent_node,
    tools=agent_tools,
    agent=visualization_generator_agent,
    name=AGENT_NAME,
)
