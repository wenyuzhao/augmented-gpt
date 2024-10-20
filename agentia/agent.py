from typing import (
    Annotated,
    Callable,
    AsyncGenerator,
    Literal,
    Optional,
    TypeVar,
    Any,
    Generic,
    overload,
    TYPE_CHECKING,
)

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


@dataclass
class UserConsentEvent:
    id: str
    message: str
    type: Literal["consent"] = "consent"


ChatCompletionEvent = ToolCallEvent | UserConsentEvent | M


@dataclass
class ChatCompletion(Generic[M]):
    def __init__(self, agen: AsyncGenerator[ChatCompletionEvent[M], None]) -> None:
        super().__init__()
        self.__agen = agen

    async def __anext__(self) -> ChatCompletionEvent[M]:
        return await self.__agen.__anext__()

    def __aiter__(self):
        return self

    async def messages(self) -> AsyncGenerator[M, None]:
        async for event in self:
            if isinstance(event, Message) or isinstance(event, MessageStream):
                yield event


AGENT_COUNTER = 0


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

        self.__backend: LLMBackend = OpenRouterBackend(
            model=model,
            tools=ToolRegistry(self, tools),
            options=options or ModelOptions(),
            instructions=instructions,
            debug=debug,
            api_key=api_key,
        )
        self.on_tool_start: Callable[[str, ToolInfo, Any], Any] | None = None
        self.on_tool_end: Callable[[str, ToolInfo, Any, Any], Any] | None = None
        self.name = name
        self.description = description
        self.colleagues: dict[str, "Agent"] = {}

        if colleagues is not None:
            self.__init_cooperation(colleagues)

    def __add_colleague(self, colleague: "Agent"):
        if colleague.name in self.colleagues:
            return
        self.colleagues[colleague.name] = colleague
        # Add a tool to dispatch a job to one colleague
        agent_names = [agent.name for agent in self.colleagues.values()]
        leader = self
        description = "Dispatch a job to a agents. Note that the agent does not have your job context, so give them the job details as precise as possible. Here are a list of agents with their description:\n"
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
            print(f"Dispatching job to {agent}: {job}")
            target = self.colleagues[agent]
            response = target.chat_completion([Message(role="user", content=job)])
            results = []
            async for message in response:
                if isinstance(message, Message):
                    print(f"Response from {agent}: {message.content}")
                    results.append(message.to_json())
            return results

        self.__backend.tools._add_dispatch_tool(dispatch_job)

    def __init_cooperation(self, colleagues: list["Agent"]):
        # Leader can dispatch jobs to colleagues
        for colleague in colleagues:
            self.__add_colleague(colleague)
        # Colleagues can communicate with each other
        for i in range(len(colleagues)):
            for j in range(i + 1, len(colleagues)):
                colleagues[i].__add_colleague(colleagues[j])
                colleagues[j].__add_colleague(colleagues[i])

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
