import boto3
import textwrap

region_name = "ap-south-1"  # make sure this is the same region as the region where you created your agent


def invoke_agent(
    agentId: str,
    agentAliasId: str,
    inputText: str,
    sessionId: str,
    enableTrace: bool = False,
    endSession: bool = False,
    width: int = 70,
):
    """
    Invokes a Bedrock supervisor agent and optionally auto-continues the conversation using "proceed"
    when multiple agents are involved.

    Returns full response content and session ID.
    """
    session = boto3.Session(profile_name="my-session-1")

    bedrock_agent_runtime = session.client(
        "bedrock-agent-runtime",
        region_name="ap-south-1",
        aws_access_key_id=session.get_credentials().access_key,
        aws_secret_access_key=session.get_credentials().secret_key,
        aws_session_token=session.get_credentials().token,
    )

    response = bedrock_agent_runtime.invoke_agent(
        agentId=agentId,
        agentAliasId=agentAliasId,
        sessionId=sessionId,
        inputText=inputText,
        endSession=endSession,
        enableTrace=enableTrace,
    )

    event_stream = response["completion"]
    # print("Event Stream:", event_stream)
    agent_response = ""

    print(f"User: {textwrap.fill(inputText, width=width)}\n")
    print("Agent:", end=" ", flush=True)

    for event in event_stream:
        if "chunk" in event:
            chunk_text = event["chunk"].get("bytes", b"").decode("utf-8")
            if not enableTrace:  # Only print chunks if trace is not enabled
                print(
                    textwrap.fill(chunk_text, width=width, subsequent_indent="       "),
                    end="",
                    flush=True,
                )
            agent_response += chunk_text
    full_response = ""
    full_response += agent_response.strip()

    return full_response.strip()

    # while continue_convo:
    #     response = bedrock_agent_runtime.invoke_agent(
    #         agentId=agentId,
    #         agentAliasId=agentAliasId,
    #         sessionId=sessionId,
    #         inputText=current_input,
    #         endSession=endSession,
    #         enableTrace=enableTrace,
    #     )

    #     event_stream = response["completion"]
    #     agent_response = ""

    #     if current_input == inputText:
    #         print(f"User: {textwrap.fill(current_input, width=width)}\n")
    #         print("Agent:", end=" ", flush=True)
    #     else:
    #         print(f"\n[Auto] Proceeding...\n")
    #         print("Agent:", end=" ", flush=True)

    #     for event in event_stream:
    #         if "chunk" in event:
    #             chunk_text = event["chunk"].get("bytes", b"").decode("utf-8")
    #             if not enableTrace:
    #                 print(
    #                     textwrap.fill(
    #                         chunk_text, width=width, subsequent_indent="       "
    #                     ),
    #                     end="",
    #                     flush=True,
    #                 )
    #             agent_response += chunk_text
    #         elif "trace" in event and enableTrace:
    #             trace = event["trace"]
    #             trace_details = trace.get("trace", {})
    #             # You can expand trace logic here if needed for debugging or metrics

    #     full_response += agent_response.strip() + "\n"

    #     # Decide whether to proceed automatically
    #     if auto_proceed:
    #         if (
    #             "proceed" in agent_response.lower()
    #             or "do you want me to continue" in agent_response.lower()
    #         ):
    #             current_input = "proceed"
    #             continue_convo = True
    #         else:
    #             continue_convo = False
    #     else:
    #         continue_convo = False

    # print(f"\n\nSession ID: {response.get('sessionId')}")
    # return full_response.strip(), sessionId
