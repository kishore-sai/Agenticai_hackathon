def router(state):
    print("\n")
    history = state.get("messages", [])
    print("Routing based on history:", history)
    print(type(history))
    last_user_msg = history[-1] if history else ""
    last_agent = state.get("last_agent", None)
    print("Last user message:", last_user_msg)
    print("Last agent:", last_agent)

    if last_agent == "agent_1":
        print("Routing to agent_2")
        return "agent_2"
    elif last_agent == "agent_2":
        print("Routing to agent_3")
        return "agent_3"
    elif last_agent == "agent_3":
        print("Routing to agent_1")
        return "END"
    # else:
    #     print("No 'proceed' command found, routing to agent_1")
    #     return "agent_1" if last_agent is None else "END"
