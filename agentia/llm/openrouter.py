import json
import os
from typing import AsyncIterator, Literal, Any, overload, override

from .. import MSG_LOGGER

from . import LLMBackend, ModelOptions
from ..tools import ToolRegistry

from ..message import (
    AssistantMessage,
    Message,
    MessageStream,
    ToolCall,
    FunctionCall,
)
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionMessage,
)
import openai
from openai.types.chat.chat_completion_message import FunctionCall as OpenAIFunctionCall
from openai.types.beta.threads.required_action_function_tool_call import (
    Function as OpenAIThreadFunction,
)
from openai.types.chat.chat_completion_message_tool_call import (
    Function as OpenAIFunction,
)
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    ChoiceDeltaToolCall,
)
import openai
from openai.types.chat import ChatCompletionChunk


class OpenRouterBackend(LLMBackend):
    def __init__(
        self,
        model: str,
        tools: ToolRegistry,
        options: ModelOptions,
        instructions: str | None,
        api_key: str | None = None,
    ) -> None:
        super().__init__(model, tools, options, instructions)
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")
        self.client = openai.AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    @overload
    @override
    async def _chat_completion_request(
        self, messages: list[Message], stream: Literal[False]
    ) -> AssistantMessage: ...

    @overload
    @override
    async def _chat_completion_request(
        self, messages: list[Message], stream: Literal[True]
    ) -> MessageStream: ...

    @override
    async def _chat_completion_request(
        self, messages: list[Message], stream: bool
    ) -> AssistantMessage | MessageStream:
        msgs: list[ChatCompletionMessageParam] = [
            self.__message_to_ccmp(m) for m in messages
        ]
        args: Any = {
            "model": self.model,
            "messages": msgs,
            **self.options.as_kwargs(),
        }
        if not self.tools.is_empty():
            if self.support_tools():
                args["tools"] = self.tools.to_json()
                args["tool_choice"] = "auto"
            else:
                raise NotImplementedError("Functions are not supported")
        if stream:
            response = await self.client.chat.completions.create(**args, stream=True)
            return ChatMessageStream(response)
        else:
            response = await self.client.chat.completions.create(**args, stream=False)
            if response.choices is None:
                print(response)
                raise RuntimeError("response.choices is None")
            return self.__ccm_to_message(response.choices[0].message)

    def __message_to_ccmp(self, m: Message) -> ChatCompletionMessageParam:
        content = m.content or ""
        if m.role == "system":
            assert isinstance(content, str)
            return ChatCompletionSystemMessageParam(role="system", content=content)
        # if m.role == Role.FUNCTION:
        #     raise NotImplementedError("Function is not supported")
        if m.role == "tool":
            assert isinstance(content, str)
            assert m.tool_call_id is not None
            return ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id=m.tool_call_id,
                content=content,
            )
        if m.role == "user":
            _content = (
                content
                if isinstance(content, str)
                else [c.to_openai_content_part() for c in content]
            )
            return ChatCompletionUserMessageParam(role="user", content=_content)
        if m.role == "assistant":
            assert isinstance(content, str)
            if len(m.tool_calls) > 0:
                return ChatCompletionAssistantMessageParam(
                    role="assistant",
                    content=content,
                    tool_calls=[
                        self.__tool_call_to_openai_tool_call(tool_call)
                        for tool_call in m.tool_calls
                    ],
                )
            return ChatCompletionAssistantMessageParam(
                role="assistant", content=content
            )
        raise RuntimeError("Unreachable")

    def __ccm_to_message(self, m: ChatCompletionMessage) -> "AssistantMessage":
        assert m.role == "assistant"
        assert m.function_call is None
        return AssistantMessage(
            content=m.content,
            tool_calls=(
                [
                    ToolCall(
                        id=t.id,
                        function=self.__oai_function_call_to_function_call(t.function),
                        type=t.type,
                    )
                    for t in m.tool_calls
                ]
                if m.tool_calls is not None
                else []
            ),
        )

    def __tool_call_to_openai_tool_call(
        self, tool_call: ToolCall
    ) -> ChatCompletionMessageToolCallParam:
        return {
            "id": tool_call.id,
            "function": tool_call.function.to_dict(),
            "type": tool_call.type,
        }

    def __oai_function_call_to_function_call(
        self,
        x: OpenAIFunctionCall | OpenAIFunction | OpenAIThreadFunction,
    ) -> "FunctionCall":
        return FunctionCall(
            name=x.name,
            arguments=json.loads(x.arguments),
        )


class ChatMessageStream(MessageStream):
    def __init__(
        self,
        response: openai.AsyncStream[ChatCompletionChunk],
    ):
        self.__response = response
        self.__aiter = response.__aiter__()
        self.__message = AssistantMessage()
        self.__tool_calls: list[ChoiceDeltaToolCall] = []
        self.__final_message: AssistantMessage | None = None

    def __get_final_merged_tool_calls(self) -> list[ToolCall]:
        return [
            ToolCall(
                id=t.id or "",
                function=FunctionCall(
                    name=t.function.name or "",
                    arguments=json.loads(t.function.arguments or ""),
                ),
                type="function",
            )
            for t in self.__tool_calls
            if t.function is not None
        ]

    def __merge_tool_calls(self, delta: list[ChoiceDeltaToolCall]):
        for d in delta:
            if d.index is not None and d.index < len(self.__tool_calls):
                t = self.__tool_calls[d.index]
                assert t.id is not None
                t.id += d.id or ""
                assert t.function is not None
                assert d.function is not None
                t.function.name = (t.function.name or "") + (d.function.name or "")
                t.function.arguments = (t.function.arguments or "") + (
                    d.function.arguments or ""
                )
            else:
                # assert d.index == len(self.__tool_calls)
                assert d.function is not None
                self.__tool_calls.append(d)

    async def __anext__impl(self) -> str:
        if self.__final_message is not None:
            raise StopAsyncIteration()
        try:
            chunk = await self.__aiter.__anext__()
        except StopAsyncIteration:
            self.__message.tool_calls = self.__get_final_merged_tool_calls()
            self.__final_message = self.__message
            raise StopAsyncIteration()
        if hasattr(chunk, "error"):
            raise RuntimeError(chunk.error["message"])  # type: ignore
        delta = chunk.choices[0].delta
        # merge self.__message and delta
        if delta.content is not None:
            if self.__message.content is None:
                self.__message.content = ""
            assert isinstance(delta.content, str)
            assert isinstance(self.__message.content, str)
            self.__message.content += delta.content
        if delta.tool_calls is not None:
            self.__merge_tool_calls(delta.tool_calls)
        return delta.content or ""

    async def __anext__(self) -> str:
        while True:
            delta = await self.__anext__impl()
            if len(delta) > 0:
                return delta

    def __aiter__(self) -> AsyncIterator[str]:
        return self

    async def wait_for_completion(self) -> AssistantMessage:
        if self.__final_message is not None:
            return self.__final_message
        async for _ in self:
            ...
        assert self.__final_message is not None
        return self.__final_message
