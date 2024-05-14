from dataclasses import dataclass
import json
import os
from typing import (
    AsyncGenerator,
    AsyncIterator,
    Optional,
    Literal,
    Any,
    overload,
    override,
)

from .. import MSG_LOGGER

from ..llm import LLMBackend, Model, ModelOptions
from ..tools import ToolRegistry
from ..history import History
from ..message import (
    ContentPartText,
    FunctionCall,
    Message,
    Role,
    MessageStream,
    ToolCall,
)
from mistralai.async_client import MistralAsyncClient
from mistralai.models.chat_completion import (
    ChatMessage,
    ToolCall as MistralToolCall,
    FunctionCall as MistralFunctionCall,
    ChatCompletionStreamResponse,
)

from mistralai.constants import ENDPOINT, RETRY_STATUS_CODES
import httpx
from mistralai.exceptions import (
    MistralAPIException,
    MistralAPIStatusException,
    MistralConnectionException,
    MistralException,
)


class MyMistralAsyncClient(MistralAsyncClient):
    async def _check_response_status_codes(self, response: httpx.Response) -> None:
        if response.status_code in RETRY_STATUS_CODES:
            if response.stream:
                await response.aread()
            print("Retry: ", response.status_code, response.text)
            raise MistralAPIStatusException.from_response(
                response,
                message=f"Status: {response.status_code}. Message: {response.text}",
            )
        elif 400 <= response.status_code < 500:
            if response.stream:
                await response.aread()
            raise MistralAPIException.from_response(
                response,
                message=f"Status: {response.status_code}. Message: {response.text}",
            )
        elif response.status_code >= 500:
            if response.stream:
                await response.aread()
            raise MistralException(
                message=f"Status: {response.status_code}. Message: {response.text}",
            )


class MistralBackend(LLMBackend):
    def __init__(
        self,
        model: Model,
        tools: ToolRegistry,
        options: ModelOptions,
        instructions: Optional[str],
        debug: bool,
        api_key: str | None = None,
    ) -> None:
        super().__init__(model, tools, options, instructions, debug)
        api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY environment variable is not set")
        self.client = MyMistralAsyncClient(api_key=api_key, max_retries=3)

    @overload
    async def _chat_completion_request(
        self, messages: list[Message], stream: Literal[False]
    ) -> Message: ...

    @overload
    async def _chat_completion_request(
        self, messages: list[Message], stream: Literal[True]
    ) -> MessageStream: ...

    @override
    async def _chat_completion_request(
        self, messages: list[Message], stream: bool
    ) -> Message | MessageStream:
        msgs: list[ChatMessage] = [self.__message_to_cm(m) for m in messages]
        args: Any = {
            "model": self.model.model,
            "messages": msgs,
            # **self.options.as_kwargs(),
        }
        if not self.tools.is_empty():
            args["tools"] = self.tools.to_json()
            args["tool_choice"] = "auto"
        if stream:
            print(args)
            gen = self.client.chat_stream(**args, safe_mode=False, safe_prompt=False)
            return ChatMessageStream(gen)
        else:
            response = await self.client.chat(
                **args, safe_mode=False, safe_prompt=False
            )
            return self.__cm_to_message(response.choices[0].message)

    def __message_to_cm(self, m: Message) -> ChatMessage:
        content = m.content or ""
        if m.role == Role.SYSTEM:
            assert isinstance(content, str)
            return ChatMessage(role="system", content=content)
        if m.role == Role.FUNCTION:
            assert isinstance(content, str)
            assert m.name is not None
            return ChatMessage(role="function", name=m.name, content=content)
        if m.role == Role.TOOL:
            assert isinstance(content, str)
            assert m.tool_call_id is not None
            return ChatMessage(role="tool", name=m.tool_call_id, content=content)
        if m.role == Role.USER:
            if not isinstance(content, str):
                for part in content:
                    assert isinstance(
                        part, ContentPartText
                    ), "Only text content is supported"
            _content = (
                content
                if isinstance(content, str)
                else "\n".join(
                    [c.content for c in content if isinstance(c, ContentPartText)]
                )
            )
            return ChatMessage(role="user", content=_content)
        if m.role == Role.ASSISTANT:
            assert isinstance(content, str)
            if len(m.tool_calls) > 0:
                return ChatMessage(
                    role="assistant",
                    content=content,
                    tool_calls=[
                        MistralToolCall(
                            id=tool_call.id,
                            function=MistralFunctionCall(
                                name=tool_call.function.name,
                                arguments=json.dumps(tool_call.function.arguments),
                            ),
                        )
                        for tool_call in m.tool_calls
                    ],
                )
            return ChatMessage(role="assistant", content=content)
        raise RuntimeError("Unreachable")

    def __cm_to_message(self, m: ChatMessage) -> "Message":
        return Message(
            role=Role.from_str(m.role),
            content=m.content if isinstance(m.content, str) else "\n".join(m.content),
            name=m.name,
            tool_calls=(
                [
                    ToolCall(
                        id=t.id,
                        function=FunctionCall.from_dict(
                            {"name": t.function.name, "arguments": t.function.arguments}
                        ),
                        type="function",
                    )
                    for t in m.tool_calls
                ]
                if m.tool_calls is not None
                else []
            ),
        )


class ChatMessageStream(MessageStream):
    def __init__(
        self,
        generator: AsyncGenerator[ChatCompletionStreamResponse, None],
    ):
        @dataclass
        class PartialMessage:
            content: str
            tool_calls: list[ToolCall]

        async def __gen():
            partial = PartialMessage(content="", tool_calls=[])
            async for response in generator:
                print(response)
                delta = response.choices[0].delta
                partial.content += delta.content or ""
                partial.tool_calls += [
                    ToolCall(
                        id=t.id,
                        function=FunctionCall(
                            name=t.function.name,
                            arguments=json.loads(t.function.arguments),
                        ),
                        type="function",
                    )
                    for t in delta.tool_calls or []
                ]
                if delta.content is not None:
                    yield delta.content
            self.__final_message = Message(
                role=Role.ASSISTANT,
                content=partial.content,
                tool_calls=partial.tool_calls,
            )

        self.__aiter = __gen()
        self.__final_message: Optional[Message] = None

    def __aiter__(self) -> AsyncIterator[str]:
        return self.__aiter

    async def wait_for_completion(self) -> Message:
        if self.__final_message is not None:
            return self.__final_message
        async for _ in self:
            ...
        assert self.__final_message is not None
        return self.__final_message
