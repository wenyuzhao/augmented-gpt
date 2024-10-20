import asyncio
import logging
from typing import (
    Annotated,
    Callable,
    AsyncGenerator,
    Coroutine,
    Literal,
    Optional,
    TypeVar,
    Any,
    Generic,
    overload,
    TYPE_CHECKING,
)

from agentia import MSG_LOGGER

from .message import *
from .history import History

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tools import ToolInfo, ToolRegistry, Tools
    from .plugins import Plugin
    from .llm import ModelOptions

M = TypeVar("M", AssistantMessage, MessageStream)


@dataclass
class ToolCallEvent:
    agent: "Agent"
    tool: "ToolInfo"
    id: str
    function: FunctionCall
    result: Any | None = None
    type: Literal["tool"] = "tool"


ToolCallEventListener = Callable[[ToolCallEvent], Any]

DEFAULT_MODEL = "openai/gpt-4o-mini"


@dataclass
class ChatCompletion(Generic[M]):
    def __init__(self, agen: AsyncGenerator[M, None]) -> None:
        super().__init__()
        self.__agen = agen

    async def __anext__(self) -> M:
        return await self.__agen.__anext__()

    def __aiter__(self):
        return self

    async def messages(self) -> AsyncGenerator[M, None]:
        async for event in self:
            if isinstance(event, Message) or isinstance(event, MessageStream):
                yield event

    async def __await_impl(self) -> str:
        last_message = ""
        async for msg in self.__agen:
            if isinstance(msg, Message):
                assert isinstance(msg.content, str)
                last_message = msg.content
            if isinstance(msg, MessageStream):
                last_message = ""
                async for delta in msg:
                    last_message += delta
        return last_message

    def __await__(self):
        return self.__await_impl().__await__()

    async def dump(self, name: str | None = None):
        async for msg in self.__agen:
            if isinstance(msg, Message):
                if name:
                    print(f"[{name}] ", end="", flush=True)
                print(msg.content)
            if isinstance(msg, MessageStream):
                name_printed = False
                outputed = False
                async for delta in msg:
                    if delta == "":
                        continue
                    if not name_printed and name:
                        print(f"[{name}] ", end="", flush=True)
                        name_printed = True
                    outputed = True
                    print(delta, end="", flush=True)
                if outputed:
                    print()


AGENT_COUNTER = 0

UserConsentHandler = Callable[[str], bool | Coroutine[Any, Any, bool]]


class Agent:
    def __init__(
        self,
        name: str | None = None,
        description: str | None = None,
        model: Annotated[str | None, f"Default to {DEFAULT_MODEL}"] = None,
        tools: Optional["Tools"] = None,
        options: Optional["ModelOptions"] = None,
        api_key: str | None = None,
        instructions: str | None = None,
        debug: bool = False,
        colleagues: list["Agent"] | None = None,
    ):
        from .llm import LLMBackend, ModelOptions
        from .tools import ToolRegistry

        global AGENT_COUNTER
        id = AGENT_COUNTER
        AGENT_COUNTER += 1
        if name is None:
            name = f"Agent#{id}"

        self.log = MSG_LOGGER.getChild(name)
        if debug:
            self.log.setLevel(logging.DEBUG)
        model = model or DEFAULT_MODEL
        if ":" in model:
            provider = model.split(":")[0]
            model = model.split(":")[1]
        else:
            provider = "openrouter"
        if provider == "openai":
            from .llm.openai import OpenAIBackend

            self.__backend: LLMBackend = OpenAIBackend(
                model=model,
                tools=ToolRegistry(self, tools),
                options=options or ModelOptions(),
                instructions=instructions,
                api_key=api_key,
            )
        else:
            from .llm.openrouter import OpenRouterBackend

            self.__backend: LLMBackend = OpenRouterBackend(
                model=model,
                tools=ToolRegistry(self, tools),
                options=options or ModelOptions(),
                instructions=instructions,
                api_key=api_key,
            )
        self.name = name
        self.description = description
        self.colleagues: dict[str, "Agent"] = {}
        self.__user_consent_handler: UserConsentHandler | None = None
        self.__on_tool_start: Callable[[ToolCallEvent], Any] | None = None
        self.__on_tool_end: Callable[[ToolCallEvent], Any] | None = None
        self._dump_communication: bool = False
        self.context: Any = None
        self.original_config: Any = None

        if colleagues is not None and len(colleagues) > 0:
            self.__init_cooperation(colleagues)

    @staticmethod
    def set_default_model(model: str):
        global DEFAULT_MODEL
        DEFAULT_MODEL = model

    @staticmethod
    def init_logging(level: int = logging.INFO):
        """Initialize logging with a set of pre-defined rules."""
        from . import init_logging

        init_logging(level)

    async def request_for_user_consent(self, message: str) -> bool:
        if self.__user_consent_handler is not None:
            result = self.__user_consent_handler(message)
            if asyncio.iscoroutine(result):
                return await result
            return result
        return True

    def on_user_consent(self, listener: UserConsentHandler):
        self.__user_consent_handler = listener
        return listener

    def on_tool_start(self, listener: Callable[[ToolCallEvent], Any]):
        self.__on_tool_start = listener
        return listener

    def on_tool_end(self, listener: Callable[[ToolCallEvent], Any]):
        self.__on_tool_end = listener
        return listener

    async def _emit_tool_call_event(self, event: ToolCallEvent):
        async def call_listener(listener: ToolCallEventListener):
            result = listener(event)
            if asyncio.iscoroutine(result):
                await result

        if event.result is None and self.__on_tool_start is not None:
            await call_listener(self.__on_tool_start)
        if event.result is not None and self.__on_tool_end is not None:
            await call_listener(self.__on_tool_end)

    def __add_colleague(self, colleague: "Agent"):
        if colleague.name in self.colleagues:
            return
        self.colleagues[colleague.name] = colleague
        # Add a tool to dispatch a job to one colleague
        agent_names = [agent.name for agent in self.colleagues.values()]
        leader = self
        description = "Send a message or dispatch a job to a agent, and get the response from them. Note that the agent does not have any context expect what you explicitly told them, so give them the details as precise and as much as possible. Agents cannot contact each other, please coordinate the jobs and information between them properly by yourself when necessary. Here are a list of agents with their description:\n"
        for agent in self.colleagues.values():
            description += f" * {agent.name}: {agent.description}\n"

        from .decorators import tool

        @tool(name="_communiate", description=description)
        async def communiate(
            agent: Annotated[
                Annotated[str, agent_names],
                "The name of the agent to communicate with. This must be one of the provided colleague names.",
            ],
            message: Annotated[
                str, "The message to send to the agent, or the job details."
            ],
        ):
            self.log.info(f"COMMUNICATE {leader.name} -> {agent}: {repr(message)}")

            if leader._dump_communication:
                print(f"[{leader.name} -> {agent}] {message}")
            target = self.colleagues[agent]
            response = target.chat_completion(
                [
                    SystemMessage(
                        f"{leader.name} is directly talking to you right now. ({leader.name}: {leader.description})",
                    ),
                    UserMessage(message),
                ]
            )
            last_message = ""
            async for m in response:
                if isinstance(m, Message):
                    self.log.info(
                        f"RESPONSE {leader.name} <- {agent}: {repr(m.content)}"
                    )
                    # results.append(m.to_json())
                    last_message = m.content
                    if leader._dump_communication:
                        print(f"[{leader.name} <- {agent}] {m.content}")
            return last_message

        self.__backend.tools._add_dispatch_tool(communiate)

    def __init_cooperation(self, colleagues: list["Agent"]):
        # Leader can dispatch jobs to colleagues
        for colleague in colleagues:
            self.__add_colleague(colleague)
        # Colleagues can communicate with each other
        # for i in range(len(colleagues)):
        #     for j in range(i + 1, len(colleagues)):
        #         colleagues[i].__add_colleague(colleagues[j])
        #         colleagues[j].__add_colleague(colleagues[i])

    def reset(self):
        self.__backend.reset()

    def set_raw_history(self, history: Any):
        self.__backend.get_history().set_raw_messages(history)

    def get_raw_history(self) -> Any:
        return self.__backend.get_history().get_raw_messages()

    def get_plugin(self, name: str) -> "Plugin":
        return self.__backend.tools.get_plugin(name)

    @overload
    def chat_completion(
        self, messages: list[Message], stream: Literal[False] = False
    ) -> ChatCompletion[AssistantMessage]: ...

    @overload
    def chat_completion(
        self, messages: list[Message], stream: Literal[True]
    ) -> ChatCompletion[MessageStream]: ...

    def chat_completion(
        self,
        messages: list[Message],
        stream: bool = False,
    ) -> ChatCompletion[MessageStream] | ChatCompletion[AssistantMessage]:
        if stream:
            return self.__backend.chat_completion(messages, stream=True)
        else:
            return self.__backend.chat_completion(messages, stream=False)

    @property
    def history(self) -> History:
        return self.__backend.get_history()

    @property
    def model(self) -> str:
        return self.__backend.model

    @property
    def tools(self) -> "ToolRegistry":
        return self.__backend.tools

    def all_agents(self) -> set["Agent"]:
        agents = set()
        agents.add(self)
        for agent in self.colleagues.values():
            agents.update(agent.all_agents())
        return agents

    @staticmethod
    def load_from_config(
        config: str, resolver: Callable[[str], Path | None] | None = None
    ):
        from .utils.config import load_agent_from_config

        return load_agent_from_config(config, resolver)
