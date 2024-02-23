from dataclasses import dataclass
import logging
from typing import Any, List, Literal, Optional, overload

import openai

from ..tools import ToolRegistry
from ..message import Message, MessageStream
from ..augmented_gpt import ChatCompletion


GPTModel = (
    Literal[
        "gpt-4",
        "gpt-4-0613",
        "gpt-4-32k",
        "gpt-4-32k-0613",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k",
        # Preview versions
        "gpt-4-1106-preview",
        "gpt-4-vision-preview",
        "gpt-3.5-turbo-1106",
    ]
    | str
)


@dataclass
class GPTOptions:
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[dict[str, int]] = None
    max_tokens: Optional[int] = None
    n: Optional[int] = None
    presence_penalty: Optional[float] = None
    stop: Optional[str] | List[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    timeout: Optional[float] = None

    def as_kwargs(self) -> dict[str, Any]:
        args = {
            "frequency_penalty": self.frequency_penalty,
            "logit_bias": self.logit_bias,
            "max_tokens": self.max_tokens,
            "n": self.n,
            "presence_penalty": self.presence_penalty,
            "stop": self.stop,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "timeout": self.timeout,
        }
        return {k: v for k, v in args.items() if v is not None}


class ChatGPTBackend:
    def support_tools(self) -> bool:
        return "vision" not in self.model

    def __init__(
        self,
        model: GPTModel,
        tools: ToolRegistry,
        gpt_options: GPTOptions,
        api_key: str,
        prologue: list[Message],
    ):
        self.gpt_options = gpt_options or GPTOptions()
        self.model = model
        self.api_key = api_key
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.logger = logging.getLogger("AugmentedGPT")
        # self.logger.setLevel(logging.DEBUG if debug else logging.INFO)
        self.tools = tools
        self.__prologue = prologue or []
        self.history: list[Message] = [m for m in self.__prologue] or []

    def reset(self):
        self.history = [m for m in self.__prologue]

    @overload
    def chat_completion(
        self,
        messages: List[Message],
        stream: Literal[False] = False,
        context_free: bool = False,
    ) -> ChatCompletion[Message]: ...

    @overload
    def chat_completion(
        self, messages: List[Message], stream: Literal[True], context_free: bool = False
    ) -> ChatCompletion[Message | MessageStream]: ...

    def chat_completion(
        self,
        messages: list[Message],
        stream: bool = False,
        context_free: bool = False,
    ) -> ChatCompletion[Message | MessageStream] | ChatCompletion[Message]:
        raise NotImplementedError

    async def _on_new_chat_message(self, msg: Message):
        await self.tools.on_new_chat_message(msg)
