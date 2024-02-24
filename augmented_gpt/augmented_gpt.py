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
from .tools import ToolRegistry, Tools
import os

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .plugins import Plugin
    from .gpt import GPTModel, GPTOptions

M = TypeVar("M", Message, MessageStream)


class ChatCompletion(Generic[M]):
    def __init__(self, agen: AsyncGenerator[M, None]) -> None:
        super().__init__()
        self.__agen = agen

    async def __anext__(self) -> M:
        return await self.__agen.__anext__()

    def __aiter__(self):
        return self


class AugmentedGPT:
    def support_tools(self) -> bool:
        return "vision" not in self.__backend.model

    def __init__(
        self,
        model: "GPTModel" = "gpt-4-1106-preview",
        tools: Optional[Tools] = None,
        gpt_options: Optional["GPTOptions"] = None,
        api_key: Optional[str] = None,
        instructions: Optional[str] = None,
        api: Literal["chat", "assistant"] = "chat",
        name: Optional[str] = None,
        description: Optional[str] = None,
        assistant_id: str | None = None,
        thread_id: str | None = None,
        debug: bool = False,
    ):
        _api_key = api_key or os.environ.get("OPENAI_API_KEY")
        assert _api_key is not None, "Missing OPENAI_API_KEY"
        from .gpt import LLMBackend, GPTOptions
        from .gpt.chat import GPTChatBackend
        from .gpt.assistant import GPTAssistantBackend

        if api == "chat":
            self.__backend: LLMBackend = GPTChatBackend(
                model=model,
                tools=ToolRegistry(self, tools),
                gpt_options=gpt_options or GPTOptions(),
                api_key=_api_key,
                instructions=instructions,
                name=name,
                description=description,
                debug=debug,
            )
        else:
            self.__backend: LLMBackend = GPTAssistantBackend(
                model=model,
                tools=ToolRegistry(self, tools),
                gpt_options=gpt_options or GPTOptions(),
                api_key=_api_key,
                instructions=instructions,
                name=name,
                description=description,
                assistant_id=assistant_id,
                thread_id=thread_id,
                debug=debug,
            )
        self.on_tool_start: Optional[Callable[[str, str, Any], Any]] = None
        self.on_tool_end: Optional[Callable[[str, str, Any, Any], Any]] = None

    def reset(self):
        self.__backend.reset()

    @property
    def openai_client(self):
        return self.__backend.client

    def get_plugin(self, name: str) -> "Plugin":
        return self.__backend.tools.get_plugin(name)

    @overload
    def chat_completion(
        self,
        messages: List[Message],
        stream: Literal[False] = False,
    ) -> ChatCompletion[Message]: ...

    @overload
    def chat_completion(
        self, messages: List[Message], stream: Literal[True]
    ) -> ChatCompletion[MessageStream]: ...

    def chat_completion(
        self,
        messages: list[Message],
        stream: bool = False,
    ) -> ChatCompletion[MessageStream] | ChatCompletion[Message]:
        if stream:
            return self.__backend.chat_completion(
                messages,
                stream=True,
            )
        else:
            return self.__backend.chat_completion(
                messages,
                stream=False,
            )

    def get_current_assistant_id(self) -> Optional[str]:
        return self.__backend.get_current_assistant_id()

    def get_current_thread_id(self) -> Optional[str]:
        return self.__backend.get_current_thread_id()
