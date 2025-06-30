import os
from typing import Literal

from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from pydantic import BaseModel

from datachat.agents.context_finder.agent import AGENT_NAME as CONTEXT_FINDER_AGENT_NAME
from datachat.agents.execute_query.agent import AGENT_NAME as SQL_EXECUTOR_AGENT_NAME
from datachat.agents.generate_plot.agent import (
    AGENT_NAME as VISUALIZATION_GENERATOR_AGENT_NAME,
)
from datachat.agents.generate_query.agent import AGENT_NAME as SQL_GENERATOR_AGENT_NAME
from datachat.agents.helpers import get_llm
from datachat.agents.validate_query.agent import AGENT_NAME as SQL_VALIDATOR_AGENT_NAME

members_with_description = {
    "contextFinder": "Generates a list of relevant context documents for the given question",
    "sqlGenerator": "Generates a SQL Query from the given list of relevant context documents and question",
    "sqlValidator": "Validates a SQL Query based on the context documents",
    "sqlExecutor": "Executes the given SQL Query and generates a dataframe from the query results",
    "visualizationGenerator": "Generates a chart from the dataframe results",
}

members = [
    CONTEXT_FINDER_AGENT_NAME,
    SQL_GENERATOR_AGENT_NAME,
    SQL_VALIDATOR_AGENT_NAME,
    SQL_EXECUTOR_AGENT_NAME,
    VISUALIZATION_GENERATOR_AGENT_NAME,
]
LANGCHAIN_TRACING_V2 = os.environ["LANGCHAIN_TRACING_V2"]
LANGCHAIN_API_KEY = os.environ["LANGCHAIN_API_KEY"]
LANGCHAIN_ENDPOINT = os.environ["LANGCHAIN_ENDPOINT"]
LANGCHAIN_PROJECT = os.environ["LANGCHAIN_PROJECT"]

llm = get_llm()

system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers: {members_with_description}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. If required, you can pass on the result of one worker to the other."
    " When fully finished, respond with FINISH."
)
# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = members + ["FINISH"]


class routeResponse(BaseModel):
    next: Literal[*options]


route_parser = PydanticOutputParser(pydantic_object=routeResponse)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "human",
            "Given the question above, you need to work with {members} to generate the relevant context documents and a sql query from the context documents, validate the query based on context documents, execute it and generate a chart for the output. "
            "Once the chart is generated, consider the task as finished."
            "\nNow, select one of {options} to act next."
            "\n{format_instructions}"
            "\nOutput should conform to the above format instructions. DO NOT REASON THE OUTPUT.",
        ),
    ]
).partial(
    options=", ".join(options),
    members=", ".join(members),
    members_with_description=", ".join(
        [f"{key}- {value}" for key, value in members_with_description.items()]
    ),
    format_instructions=route_parser.get_format_instructions(),
)

output_fixing_prompt = PromptTemplate(
    template="""Instructions:
        --------------
        {instructions}
        --------------
        Completion:
        --------------
        {completion}
        --------------

        Above, the Completion did not satisfy the constraints given in the Instructions.
        Error:
        --------------
        {error}
        --------------

        Please try again. Please only respond with an answer that satisfies the constraints laid out in the Instructions:
        Please modify the response conform to the above output.
        For example, if the tool name is codeGenerator the output should be "{{next:codeGenerator}}"
    """
)

response_parser = OutputFixingParser.from_llm(
    llm=llm, parser=route_parser, max_retries=3, prompt=output_fixing_prompt
)


def supervisor_agent(state):
    print("===" * 5, "SUPERVISOR PROMPT BEGINS", "===" * 5)
    print(prompt)
    print("===" * 5, "SUPERVISOR PROMPT ENDS", "===" * 5)
    supervisor_chain = prompt | llm | response_parser
    output = supervisor_chain.invoke(state)
    return output
