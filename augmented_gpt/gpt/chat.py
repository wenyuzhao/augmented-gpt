from typing import (
    AsyncGenerator,
    Literal,
    Any,
    overload,
)

from augmented_gpt.augmented_gpt import ChatCompletion
from augmented_gpt.gpt import ChatGPTBackend, GPTModel, GPTOptions
from augmented_gpt.tools import ToolRegistry

from ..message import *
from openai.types.chat import ChatCompletionMessageParam


class ChatBackend(ChatGPTBackend):
    def __init__(
        self,
        model: GPTModel,
        tools: ToolRegistry,
        gpt_options: GPTOptions,
        api_key: str,
        prologue: list[Message],
        name: Optional[str],
        description: Optional[str],
        debug: bool,
    ) -> None:
        super().__init__(
            model, tools, gpt_options, api_key, prologue, name, description, debug
        )
        self.history: list[Message] = [m for m in self._prologue] or []

    def reset(self):
        self.history = [m for m in self._prologue]

    @overload
    async def __chat_completion_request(
        self, messages: list[Message], stream: Literal[False]
    ) -> Message: ...

    @overload
    async def __chat_completion_request(
        self, messages: list[Message], stream: Literal[True]
    ) -> MessageStream: ...

    async def __chat_completion_request(
        self, messages: list[Message], stream: bool
    ) -> Message | MessageStream:
        msgs: list[ChatCompletionMessageParam] = [
            m.to_chat_completion_message_param() for m in messages
        ]
        args: Any = {
            "model": self.model,
            "messages": msgs,
            **self.gpt_options.as_kwargs(),
        }
        if not self.tools.is_empty():
            if self.support_tools():
                args["tools"] = self.tools.to_json()
                args["tool_choice"] = "auto"
            else:
                args["functions"] = self.tools.to_json(legacy=True)
                args["function_call"] = "auto"
        if stream:
            response = await self.client.chat.completions.create(**args, stream=True)
            return ChatMessageStream(response)
        else:
            response = await self.client.chat.completions.create(**args, stream=False)
            return Message.from_chat_completion_message(response.choices[0].message)

    @overload
    async def __chat_completion(
        self,
        messages: list[Message],
        stream: Literal[False] = False,
        context_free: bool = False,
    ) -> AsyncGenerator[Message, None]: ...

    @overload
    async def __chat_completion(
        self, messages: list[Message], stream: Literal[True], context_free: bool = False
    ) -> AsyncGenerator[MessageStream, None]: ...

    async def __chat_completion(
        self,
        messages: list[Message],
        stream: bool = False,
        context_free: bool = False,
    ):
        history = [h for h in (self.history if not context_free else [])]
        old_history_length = len(history)
        history.extend(messages)
        for m in messages:
            await self._on_new_chat_message(m)
        # First completion request
        message: Message
        if stream:
            s = await self.__chat_completion_request(history, stream=True)
            yield s
            message = await s.wait_for_completion()
        else:
            message = await self.__chat_completion_request(history, stream=False)
            yield message
        history.append(message)
        await self._on_new_chat_message(message)
        while message.function_call is not None or len(message.tool_calls) > 0:
            if len(message.tool_calls) > 0:
                for t in message.tool_calls:
                    assert t.type == "function"
                    result = await self.tools.call_function(t.function, tool_id=t.id)
                    history.append(result)
                    # await self._on_new_chat_message(result)
                    yield result
            else:
                assert message.function_call is not None
                # ChatGPT wanted to call a user-defined function
                result = await self.tools.call_function(
                    message.function_call, tool_id=None
                )
                history.append(result)
                # await self._on_new_chat_message(result)
                yield result
            # Send back the function call result
            message: Message
            if stream:
                r = await self.__chat_completion_request(history, stream=True)
                yield r
                message = await r.wait_for_completion()
            else:
                message = await self.__chat_completion_request(history, stream=False)
                yield message
            history.append(message)
            await self._on_new_chat_message(message)
        if not context_free:
            self.history.extend(history[old_history_length:])

    @overload
    def chat_completion(
        self,
        messages: list[Message],
        stream: Literal[False] = False,
        context_free: bool = False,
    ) -> ChatCompletion[Message]: ...

    @overload
    def chat_completion(
        self, messages: list[Message], stream: Literal[True], context_free: bool = False
    ) -> ChatCompletion[MessageStream]: ...

    def chat_completion(
        self,
        messages: list[Message],
        stream: bool = False,
        context_free: bool = False,
    ) -> ChatCompletion[MessageStream] | ChatCompletion[Message]:
        if stream:
            return ChatCompletion(
                self.__chat_completion(messages, stream=True, context_free=context_free)
            )
        else:
            return ChatCompletion(
                self.__chat_completion(
                    messages, stream=False, context_free=context_free
                )
            )
