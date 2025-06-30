from agents.invoke_agent import invoke_agent
import uuid


def agent_node(state):
    # Use provided session ID if it exists in state, else generate a new one
    session_id = state.get("sessionId", str(uuid.uuid4()))

    input_text = None
    messages = state.get("messages", [])
    # print(f"Current messages in state: {messages}")
    if messages:
        input_text = messages[-1].content  # Assume last user message is the query
    else:
        input_text = "How long do patients usually wait before getting treated in the emergency room?"

    # print(f"Input text for intent agent: {input_text}")
    # print(type(input_text))

    response = invoke_agent(
        agentId="7R1CGW8POJ",
        agentAliasId="KREBUOLE26",
        inputText=input_text,
        sessionId=session_id,
    )

    return {
        "output": response,
        "sessionId": session_id,
        "last_agent": "agent_1",
        "messages": messages + [response],
    }
