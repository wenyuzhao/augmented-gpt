import asyncio
from agentia.agent import Agent, ToolCallEvent, CommunicationEvent
from agentia.message import Message, UserMessage
from agentia.utils.config import load_agent_from_config
import rich


async def __run_async(agent: Agent):
    await agent.init()
    while True:
        try:
            console = rich.console.Console()
            prompt = console.input("[bold green]>[/bold green] ").strip()
            if prompt == "exit" or prompt == "quit":
                break
        except EOFError:
            break
        response = agent.chat_completion([UserMessage(prompt)], stream=True)
        await response.dump()


def run(agent: Agent | str):
    if isinstance(agent, str):
        agent = load_agent_from_config(agent)

    def tool_start(e: ToolCallEvent):
        agent = f"{e.agent.icon} {e.agent.name}" if e.agent.icon else f"{e.agent.name}"
        tool = e.tool.display_name
        if tool == "_communiate":
            return
        rich.print(f"[bold magenta][{agent}][/bold magenta] [magenta]{tool}[/magenta]")

    def communication_start(e: CommunicationEvent):
        p, c = e.parent, e.child
        from_agent = f"{p.icon} {p.name}" if p.icon else f"{p.name}"
        to_agent = f"{c.icon} {c.name}" if c.icon else f"{c.name}"
        rich.print(
            f"[bold magenta][{from_agent} -> {to_agent}][/bold magenta] [bright_black]{e.message}[/bright_black]"
        )

    def communication_end(e: CommunicationEvent):
        p, c = e.parent, e.child
        from_agent = f"{p.icon} {p.name}" if p.icon else f"{p.name}"
        to_agent = f"{c.icon} {c.name}" if c.icon else f"{c.name}"
        rich.print(
            f"[bold magenta][{from_agent} <- {to_agent}][/bold magenta] [bright_black]{e.response}[/bright_black]"
        )

    all_agents = agent.all_agents()
    for a in all_agents:
        a.on_tool_start(tool_start)
        a.on_commuication_start(communication_start)
        a.on_commuication_end(communication_end)

    asyncio.run(__run_async(agent))
