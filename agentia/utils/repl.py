import asyncio
from agentia.agent import Agent, ToolCallEvent
from agentia.message import Message, UserMessage
from agentia.utils.config import load_agent_from_config


async def run_async(agent: Agent):
    agent._dump_communication = True
    while True:
        try:
            prompt = input("> ").strip()
            if prompt == "exit" or prompt == "quit":
                break
        except EOFError:
            break
        response = agent.chat_completion([UserMessage(prompt)], stream=True)
        await response.dump(name=agent.name)


def run(agent: Agent | str):
    if isinstance(agent, str):
        agent = load_agent_from_config(agent)

    def tool_start(e: ToolCallEvent):
        print(f"[{e.agent.name}] -> {e.tool.display_name}")

    all_agents = agent.all_agents()
    for a in all_agents:
        a.on_tool_start(tool_start)

    asyncio.run(run_async(agent))
