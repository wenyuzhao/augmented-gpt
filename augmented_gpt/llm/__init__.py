from dataclasses import dataclass
from typing import Any, AsyncGenerator, List, Literal, Optional, overload

from ..tools import ToolRegistry
from ..message import FunctionCall, Message, MessageStream
from ..augmented_gpt import (
    ChatCompletion,
    ChatCompletionEvent,
    UserConsentEvent,
    ToolCallEvent,
)
from ..history import History

from dataclasses import dataclass
from .. import MSG_LOGGER


@dataclass
class ModelOptions:
    frequency_penalty: Optional[float] = None
    # logit_bias: Optional[dict[str, int]] = None
    # max_tokens: Optional[int] = None
    # n: Optional[int] = None
    presence_penalty: Optional[float] = None
    # stop: Optional[str] | List[str] = None
    temperature: Optional[float] = None
    # repetition_penalty: Optional[float] = None
    # top_p: Optional[float] = None
    # timeout: Optional[float] = None

    def as_kwargs(self) -> dict[str, Any]:
        args = {
            "frequency_penalty": self.frequency_penalty,
            # "logit_bias": self.logit_bias,
            # "max_tokens": self.max_tokens,
            # "n": self.n,
            "presence_penalty": self.presence_penalty,
            # "stop": self.stop,
            "temperature": self.temperature,
            # "repetition_penalty": self.repetition_penalty,
            # "top_p": self.top_p,
            # "timeout": self.timeout,
        }
        return {k: v for k, v in args.items() if v is not None}


class LLMBackend:
    def support_tools(self) -> bool:
        return True

    def __init__(
        self,
        model: str,
        tools: ToolRegistry,
        options: ModelOptions,
        instructions: Optional[str],
        debug: bool,
    ):
        self.debug = debug
        self.options = options or ModelOptions()
        self.model = model
        self.tools = tools
        self.instructions = instructions
        self.history = History(instructions=instructions)

    def reset(self):
        """Clear and reset all history"""
        self.history.reset()

    def get_history(self) -> History:
        return self.history

    @overload
    def chat_completion(
        self, messages: List[Message], stream: Literal[False] = False
    ) -> ChatCompletion[Message]: ...

    @overload
    def chat_completion(
        self, messages: List[Message], stream: Literal[True]
    ) -> ChatCompletion[MessageStream]: ...

    def chat_completion(
        self, messages: list[Message], stream: bool = False
    ) -> ChatCompletion[MessageStream] | ChatCompletion[Message]:
        if stream:
            return ChatCompletion(self.__chat_completion(messages, stream=True))
        else:
            return ChatCompletion(self.__chat_completion(messages, stream=False))

    async def _on_new_chat_message(self, msg: Message):
        await self.tools.on_new_chat_message(msg)

    @overload
    async def _chat_completion_request(
        self, messages: list[Message], stream: Literal[False]
    ) -> Message: ...

    @overload
    async def _chat_completion_request(
        self, messages: list[Message], stream: Literal[True]
    ) -> MessageStream: ...

    async def _chat_completion_request(
        self, messages: list[Message], stream: bool
    ) -> Message | MessageStream:
        raise NotImplementedError

    @overload
    async def __chat_completion(
        self, messages: list[Message], stream: Literal[False] = False
    ) -> AsyncGenerator[ChatCompletionEvent[Message], None]: ...

    @overload
    async def __chat_completion(
        self, messages: list[Message], stream: Literal[True] = True
    ) -> AsyncGenerator[ChatCompletionEvent[MessageStream], None]: ...

    async def __chat_completion(self, messages: list[Message], stream: bool = False):
        history = [h for h in self.history.get()]
        old_history_length = len(history)
        history.extend(messages)
        for m in messages:
            MSG_LOGGER.info(f"{m}")
            await self._on_new_chat_message(m)
        # First completion request
        message: Message
        if stream:
            s = await self._chat_completion_request(history, stream=True)
            yield s
            message = await s.wait_for_completion()
        else:
            message = await self._chat_completion_request(history, stream=False)
            if message.content is not None:
                yield message
        history.append(message)
        MSG_LOGGER.info(f"{message}")
        await self._on_new_chat_message(message)
        # Run tools and submit results until convergence
        while len(message.tool_calls) > 0:
            # Run tools
            async for event in self.tools.call_tools(message.tool_calls):
                if isinstance(event, Message):
                    history.append(event)
                else:
                    yield event
            # Submit results
            message: Message
            if stream:
                r = await self._chat_completion_request(history, stream=True)
                yield r
                message = await r.wait_for_completion()
            else:
                message = await self._chat_completion_request(history, stream=False)
                if message.content is not None:
                    yield message
            history.append(message)
            MSG_LOGGER.info(f"{message}")
            await self._on_new_chat_message(message)
        if _unreachable:
            yield UserConsentEvent(id="", message="")
        for h in history[old_history_length:]:
            self.history.add(h)


_unreachable = False
