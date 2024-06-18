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
import os

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tools import ToolInfo, ToolRegistry, Tools
    from .plugins import Plugin
    from .llm import Model, ModelOptions

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


class AugmentedGPT:
    def support_tools(self) -> bool:
        return True

    def __init__(
        self,
        model: Union["Model", str] = "gpt-4-turbo",
        tools: Optional["Tools"] = None,
        options: Optional["ModelOptions"] = None,
        api_key: Optional[str] = None,
        instructions: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        debug: bool = False,
    ):
        from .llm import LLMBackend, ModelOptions, Model
        from .llm.openai import OpenAIBackend
        from .tools import ToolRegistry

        if isinstance(model, str):
            model = Model(model)
        if model.api == "openai":
            self.__backend: LLMBackend = OpenAIBackend(
                model=model,
                tools=ToolRegistry(self, tools),
                options=options or ModelOptions(),
                instructions=instructions,
                debug=debug,
                api_key=api_key,
            )
        else:
            raise NotImplemented
        self.on_tool_start: Optional[Callable[[str, ToolInfo, Any], Any]] = None
        self.on_tool_end: Optional[Callable[[str, ToolInfo, Any, Any], Any]] = None
        self.name = name
        self.description = description

    def reset(self):
        self.__backend.reset()

    # @property
    # def openai_client(self):
    #     return self.__backend.client

    def get_plugin(self, name: str) -> "Plugin":
        return self.__backend.tools.get_plugin(name)

    @overload
    def chat_completion(
        self,
        messages: List[Message],
        stream: Literal[False] = False,
        context: Any = None,
    ) -> ChatCompletion[Message]: ...

    @overload
    def chat_completion(
        self, messages: List[Message], stream: Literal[True]
    ) -> ChatCompletion[MessageStream]: ...

    def chat_completion(
        self,
        messages: list[Message],
        stream: bool = False,
        context: Any = None,
    ) -> ChatCompletion[MessageStream] | ChatCompletion[Message]:
        if stream:
            return self.__backend.chat_completion(
                messages,
                stream=True,
                context=context,
            )
        else:
            return self.__backend.chat_completion(
                messages,
                stream=False,
                context=context,
            )

    def get_history(self) -> History:
        return self.__backend.get_history()

    def get_model(self) -> "Model":
        return self.__backend.model

    def get_tools(self) -> "ToolRegistry":
        return self.__backend.tools
