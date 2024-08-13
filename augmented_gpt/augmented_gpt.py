from typing import (
    Callable,
    AsyncGenerator,
    List,
    Literal,
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


class AugmentedGPT:
    def support_tools(self) -> bool:
        return True

    def __init__(
        self,
        model: str = "openai/gpt-4o",
        tools: Optional["Tools"] = None,
        options: Optional["ModelOptions"] = None,
        api_key: Optional[str] = None,
        instructions: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        debug: bool = False,
    ):
        from .llm import LLMBackend, ModelOptions
        from .llm.openrouter import OpenRouterBackend
        from .tools import ToolRegistry

        self.__backend: LLMBackend = OpenRouterBackend(
            model=model,
            tools=ToolRegistry(self, tools),
            options=options or ModelOptions(),
            instructions=instructions,
            debug=debug,
            api_key=api_key,
        )
        self.on_tool_start: Optional[Callable[[str, ToolInfo, Any], Any]] = None
        self.on_tool_end: Optional[Callable[[str, ToolInfo, Any, Any], Any]] = None
        self.name = name
        self.description = description

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
        self, messages: List[Message], stream: Literal[False] = False
    ) -> ChatCompletion[Message]: ...

    @overload
    def chat_completion(
        self, messages: List[Message], stream: Literal[True] = True
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
