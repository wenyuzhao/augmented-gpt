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

M = TypeVar("M", Message, MessageStream)


@dataclass
class ToolCallEvent:
    id: str
    function: FunctionCall
    result: Any | None = None
    type: Literal["tool"] = "tool"


ToolCallEventListener = Callable[[ToolCallEvent], Any]


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

    async def dump(self):
        async for msg in self.__agen:
            if isinstance(msg, Message):
                print(msg.content)
            if isinstance(msg, MessageStream):
                async for delta in msg:
                    print(delta, end="")
                print()


AGENT_COUNTER = 0

UserConsentHandler = Callable[[str], bool | Coroutine[Any, Any, bool]]


class Agent:
    def support_tools(self) -> bool:
        return True

    def __init__(
        self,
        name: str | None = None,
        description: str | None = None,
        model: str = "openai/gpt-4o",
        tools: Optional["Tools"] = None,
        options: Optional["ModelOptions"] = None,
        api_key: str | None = None,
        instructions: str | None = None,
        debug: bool = False,
        colleagues: list["Agent"] | None = None,
    ):
        from .llm import LLMBackend, ModelOptions
        from .llm.openrouter import OpenRouterBackend
        from .tools import ToolRegistry

        global AGENT_COUNTER
        id = AGENT_COUNTER
        AGENT_COUNTER += 1
        if name is None:
            name = f"Agent#{id}"

        self.log = MSG_LOGGER.getChild(name)
        if debug:
            self.log.setLevel(logging.DEBUG)
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

        if colleagues is not None:
            self.__init_cooperation(colleagues)

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
        description = "Dispatch a job to a agent. Note that the agent does not have any context expect what you explicitly told them, so give them the job details as precise and as much as possible. Agents cannot contact each other, please coordinate the jobs between them properly by yourself to complete the whole task. Here are a list of agents with their description:\n"
        for agent in self.colleagues.values():
            description += f" * {agent.name}: {agent.description}\n"

        from .decorators import tool

        @tool(description=description)
        async def dispatch_job(
            agent: Annotated[
                Annotated[str, agent_names],
                "The name of the agent to dispatch the job to.",
            ],
            job: Annotated[str, "The job to ask the agent to do."],
        ):
            self.log.info(f"DISPATCH {leader.name} -> {agent}: {repr(job)}")
            target = self.colleagues[agent]
            job_message = f"This is a job assigned to you by {leader.name} ({leader.description}):\n\n{job}"
            response = target.chat_completion(
                [Message(role="user", content=job_message)]
            )
            results = []
            async for message in response:
                if isinstance(message, Message):
                    self.log.info(
                        f"RESPONSE {leader.name} <- {agent}: {repr(message.content)}"
                    )
                    results.append(message.to_json())
            return results

        self.__backend.tools._add_dispatch_tool(dispatch_job)

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
    ) -> ChatCompletion[Message]: ...

    @overload
    def chat_completion(
        self, messages: list[Message], stream: Literal[True]
    ) -> ChatCompletion[MessageStream]: ...

    def chat_completion(
        self,
        messages: list[Message],
        stream: bool = False,
    ) -> ChatCompletion[MessageStream] | ChatCompletion[Message]:
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
