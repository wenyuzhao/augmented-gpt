from typing import (
    AsyncGenerator,
    Literal,
    Any,
    overload,
)
from augmented_gpt import MSG_LOGGER

from augmented_gpt.augmented_gpt import ChatCompletion
from augmented_gpt.gpt import LLMBackend, GPTModel, GPTOptions
from augmented_gpt.tools import ToolRegistry

from ..message import *
from openai.types.chat import ChatCompletionMessageParam


class GPTChatBackend(LLMBackend):
    def __init__(
        self,
        model: GPTModel,
        tools: ToolRegistry,
        gpt_options: GPTOptions,
        api_key: str,
        instructions: Optional[str],
        name: Optional[str],
        description: Optional[str],
        debug: bool,
    ) -> None:
        super().__init__(
            model, tools, gpt_options, api_key, instructions, name, description, debug
        )
        self.history = History(instructions=instructions)

    def reset(self):
        self.history.reset()

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
        self, messages: list[Message], stream: Literal[False] = False
    ) -> AsyncGenerator[Message, None]: ...

    @overload
    async def __chat_completion(
        self, messages: list[Message], stream: Literal[True]
    ) -> AsyncGenerator[MessageStream, None]: ...

    async def __chat_completion(
        self,
        messages: list[Message],
        stream: bool = False,
    ):
        history = [h for h in self.history.get()]
        old_history_length = len(history)
        history.extend(messages)
        for m in messages:
            MSG_LOGGER.info(f"{m}")
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
        MSG_LOGGER.info(f"{message}")
        await self._on_new_chat_message(message)
        # Run tools and submit results until convergence
        while message.function_call is not None or len(message.tool_calls) > 0:
            # Run tools
            if len(message.tool_calls) > 0:
                results = await self.tools.call_tools(message.tool_calls)
                history.extend(results)
            else:
                assert message.function_call is not None
                r = await self.tools.call_function(message.function_call, tool_id=None)
                history.append(r)
            # Submit results
            message: Message
            if stream:
                r = await self.__chat_completion_request(history, stream=True)
                yield r
                message = await r.wait_for_completion()
            else:
                message = await self.__chat_completion_request(history, stream=False)
                yield message
            history.append(message)
            MSG_LOGGER.info(f"{message}")
            await self._on_new_chat_message(message)
        for h in history[old_history_length:]:
            self.history.add(h)

    @overload
    def chat_completion(
        self, messages: list[Message], stream: Literal[False] = False
    ) -> ChatCompletion[Message]: ...

    @overload
    def chat_completion(
        self, messages: list[Message], stream: Literal[True]
    ) -> ChatCompletion[MessageStream]: ...

    def chat_completion(
        self, messages: list[Message], stream: bool = False
    ) -> ChatCompletion[MessageStream] | ChatCompletion[Message]:
        if stream:
            return ChatCompletion(self.__chat_completion(messages, stream=True))
        else:
            return ChatCompletion(self.__chat_completion(messages, stream=False))


class History:
    def __init__(self, instructions: Optional[str]) -> None:
        self.__instructions = instructions
        self.__messages: list[Message] = []
        self.reset()

    def reset(self):
        self.__messages = []
        if self.__instructions is not None:
            self.add(Message(role=Role.SYSTEM, content=self.__instructions))

    def get(self) -> list[Message]:
        return self.__messages

    def add(self, message: Message):
        # TODO: auto trim history
        self.__messages.append(message)
