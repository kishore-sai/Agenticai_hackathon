import functools
import operator
import os
from typing import Annotated, Literal, Sequence, TypedDict

from langchain.agents.react.agent import create_react_agent
from langchain.chains.retrieval import create_retrieval_chain
from langchain.output_parsers import PydanticOutputParser
from langchain_core.agents import AgentAction
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from llm import QueryAndDescription, create_table_ddl_chain, llm
from pydantic import BaseModel
from vectordb import init_vector_store

user_question = "How many products do we have in each warehouse?"
LANCE_URI = os.environ["LANCE_URI"]
TABLE_NAME = os.environ["VECTORDB_TABLE_NAME"]

vector_store = init_vector_store(LANCE_URI, TABLE_NAME)

LANGCHAIN_TRACING_V2 = os.environ["LANGCHAIN_TRACING_V2"]
LANGCHAIN_API_KEY = os.environ["LANGCHAIN_API_KEY"]
LANGCHAIN_ENDPOINT = os.environ["LANGCHAIN_ENDPOINT"]
LANGCHAIN_PROJECT = os.environ["LANGCHAIN_PROJECT"]


# Tools
@tool
def generate_query_for_question(question):
    """Generate SQL Query based on user question.
    Parameters:
    -----------
    question: Question asked by the user

    Returns:
    --------
    A QueryAndDescription object
    """
    print("Inside Generate query function")
    parser = PydanticOutputParser(pydantic_object=QueryAndDescription)
    prompt = PromptTemplate(
        template="\n".join(
            [
                "Generate a SQL for the following question.",
                "Question: {input}",
                "\n",
                "{format_instructions}",
                "Note: The output json should be parsable by Python's json.loads function.",
                "\n",
                "Relevant table columns:",
                "{context}",
            ]
        ),
        input_variables=["input", "context"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
        output_parser=parser,
    )

    rag_chain = create_table_ddl_chain(llm=llm, prompt=prompt)
    qa_chain = create_retrieval_chain(
        retriever=vector_store.as_retriever(), combine_docs_chain=rag_chain
    )

    llm_response = qa_chain.invoke({"input": question})
    try:
        parsed_response = parser.parse(llm_response["answer"])
    except OutputParserException as oe:
        print("Handled the OutputParserException using ast.literal_eval")
        import ast

        parsed_response = QueryAndDescription.parse_obj(
            ast.literal_eval(llm_response["answer"])
        )
    return parsed_response


agent_tools = [generate_query_for_question]


def router(state) -> Literal["call_tool", "__end__", "continue"]:
    # This is the router
    messages = state["messages"]
    last_message = messages[-1]
    if "Agent execution is completed" in last_message.content:
        # Any agent decided the work is done
        return "FINISH"
    return state["next"]


# Helper Utilities
def agent_node(state, agent, name):
    global final_result
    result = agent.invoke(state)
    if isinstance(result, AgentAction):
        observation = {t.name: t for t in agent_tools}[result.tool].invoke(
            result.tool_input
        )
        final_result = observation
        return {"intermediate_steps": [(result, observation)]}
    else:
        return {"messages": [HumanMessage("Agent execution is completed")]}


# Create Agent Supervisor
members = ["sqlGenerator"]
system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)
# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = ["FINISH"] + members


class routeResponse(BaseModel):
    next: Literal[*options]


parser = PydanticOutputParser(pydantic_object=routeResponse)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}"
            "\n{format_instructions}",
        ),
    ]
).partial(
    options=str(options),
    members=", ".join(members),
    format_instructions=parser.get_format_instructions(),
)


def supervisor_agent(state):
    supervisor_chain = prompt | llm | parser
    output = supervisor_chain.invoke(state)
    return output


# The agent state is the input to each node in the graph
class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
    # The 'next' field indicates where to route to next
    next: str
    input: str


sql_agent_prompt = PromptTemplate(
    template="""
        You are a SQL developer.Your task is to generate a valid SQL statement for the natural language question.
        Don't try to execute the SQL queries.
        You have access to the following tools:
        {tools}

        ```
        Thought: you should always think about what to do. Make sure to include the previous tool results

        Action: the action to take, should be one of [{tool_names}]. Do not give me any descriptive statements, explanation or justification for picking any tool. Just give the tool name as the action.

        Action Input: the input to the action

        Critic: The critic to generate best result

        Observation: the result of the action
        ```

        ... (this Thought/Action/Action Input/Observation may repeat N times and you can call the same tool multiple times if you are not satisfied with the output.)

        When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

        ```
        Thought: I now know the final answer

        Final Answer: the final answer to the original input question
        ```

        Begin!

        Input question:{input}
        Thought: {agent_scratchpad}

        Check your output and make sure it conforms! Do not output an action and a final answer at the same time.
        """
)

sql_generator_agent = create_react_agent(
    llm, tools=agent_tools, prompt=sql_agent_prompt
)
sql_generator_node = functools.partial(
    agent_node, agent=sql_generator_agent, name="sqlGenerator"
)
# visualization_agent_prompt = PromptTemplate(
#     template="""
#         You are a SQL developer.Your task is to generate a valid SQL statement for the natural language question.
#         Don't try to execute the SQL queries.
#         You have access to the following tools:
#         {tools}

#         ```
#         Thought: you should always think about what to do. Make sure to include the previous tool results

#         Action: the action to take, should be one of [{tool_names}]. Do not give me any descriptive statements, explanation or justification for picking any tool. Just give the tool name as the action.

#         Action Input: the input to the action

#         Critic: The critic to generate best result

#         Observation: the result of the action
#         ```

#         ... (this Thought/Action/Action Input/Observation may repeat N times and you can call the same tool multiple times if you are not satisfied with the output.)

#         When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

#         ```
#         Thought: I now know the final answer

#         Final Answer: the final answer to the original input question
#         ```

#         Begin!

#         Input question:{input}
#         Thought: {agent_scratchpad}

#         Check your output and make sure it conforms! Do not output an action and a final answer at the same time.
#         """
# )
# visualization_agent = create_react_agent(llm, tools=agent_tools, prompt=visualization_agent_prompt)
# visualization_generator_node = functools.partial(
#     agent_node, agent=sql_generator_agent, name="plotGenerator"
# )
workflow = StateGraph(AgentState)
workflow.add_node("sqlGenerator", sql_generator_node)
# workflow.add_node("plotGenerator", visualization_generator_node)
workflow.add_node("supervisor", supervisor_agent)

for member in members:
    # We want our workers to ALWAYS "report back" to the supervisor when done
    workflow.add_edge(member, "supervisor")
# The supervisor populates the "next" field in the graph state
# which routes to a node or finishes
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", router, conditional_map)
# Finally, add entrypoint
workflow.add_edge(START, "supervisor")

graph = workflow.compile()

# graph.get_graph().draw_png('image.png')

result_var = graph.invoke(
    {
        "messages": [
            HumanMessage(content="How many products do we have in each warehouse?")
        ],
        "input": user_question,
    }
)

print("Out")
# print(result_var['intermediate_steps'][-1][1])
print(final_result)
