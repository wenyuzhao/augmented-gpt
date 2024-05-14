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

from ..augmented_gpt import ChatCompletion
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

# api_key = os.environ["MISTRAL_API_KEY"]
# model = "mistral-large-latest"

# client = MistralClient(api_key=api_key)

# chat_response = client.chat(
#     model=model,
#     messages=[ChatMessage(role="user", content="What is the best French cheese?")],
# )

# print(chat_response.choices[0].message.content)


class MistralBackend(LLMBackend):
    def __init__(
        self,
        model: Model,
        tools: ToolRegistry,
        options: ModelOptions,
        instructions: Optional[str],
        debug: bool,
    ) -> None:
        super().__init__(model, tools, options, instructions, debug)
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY environment variable is not set")
        self.history = History(instructions=instructions)
        self.client = MistralAsyncClient(api_key=api_key)

    def reset(self):
        self.history.reset()

    def __message_to_chat_message_param(self, m: Message) -> ChatMessage:
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
            _content = (
                content
                if isinstance(content, str)
                else [c.content for c in content if isinstance(c, ContentPartText)]
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
        msgs: list[ChatMessage] = [
            self.__message_to_chat_message_param(m) for m in messages
        ]
        args: Any = {
            "model": self.model.model,
            "messages": msgs,
            **self.options.as_kwargs(),
        }
        if not self.tools.is_empty():
            if self.support_tools():
                args["tools"] = self.tools.to_json()
                args["tool_choice"] = "auto"
            else:
                args["functions"] = self.tools.to_json(legacy=True)
                args["function_call"] = "auto"
        if stream:
            gen = self.client.chat_stream(**args)
            return ChatMessageStream(gen)
        else:
            response = await self.client.chat(**args)
            return self.__cm_to_message(response.choices[0].message)

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
        self.__generator = generator
        self.__aiter = generator.__aiter__()
        self.__message = Message(role=Role.ASSISTANT)
        self.__tool_calls: list[ToolCall] = []
        self.__final_message: Optional[Message] = None

    def __merge_tool_calls(self, delta: list[MistralToolCall]):
        self.__tool_calls += [
            ToolCall(
                id=t.id,
                function=FunctionCall(
                    name=t.function.name,
                    arguments=json.loads(t.function.arguments),
                ),
                type="function",
            )
            for t in delta
        ]

    async def __anext__impl(self) -> str:
        if self.__final_message is not None:
            raise StopAsyncIteration()
        try:
            chunk = await self.__aiter.__anext__()
        except StopAsyncIteration:
            self.__message.tool_calls = self.__tool_calls
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
        if delta.role is not None:
            self.__message.role = Role.from_str(delta.role)
        return delta.content or ""

    async def __anext__(self) -> str:
        while True:
            delta = await self.__anext__impl()
            if len(delta) > 0:
                return delta

    def __aiter__(self) -> AsyncIterator[str]:
        return self

    async def wait_for_completion(self) -> Message:
        if self.__final_message is not None:
            return self.__final_message
        async for _ in self:
            ...
        assert self.__final_message is not None
        return self.__final_message
