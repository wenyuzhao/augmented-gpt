from typing import (
    AsyncIterator,
    Literal,
    Optional,
    TypeAlias,
    Union,
    cast,
    Mapping,
    Sequence,
    Any,
)
import json
from dataclasses import dataclass, field
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    ChoiceDeltaToolCall,
)
import openai
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionFunctionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionMessage,
    ChatCompletionContentPartTextParam,
    ChatCompletionContentPartImageParam,
)

from openai.types.chat.chat_completion_message import FunctionCall as OpenAIFunctionCall
from openai.types.chat.chat_completion_assistant_message_param import (
    FunctionCall as OpenAIFunctionCallDict,
)
from openai.types.chat.chat_completion_message_tool_call import (
    Function as OpenAIFunction,
)
from enum import StrEnum

JSON: TypeAlias = (
    Mapping[str, "JSON"] | Sequence["JSON"] | str | int | float | bool | None
)


@dataclass
class FunctionCall:
    name: str
    arguments: JSON

    def to_openai_func_call_dict(self) -> OpenAIFunctionCallDict:
        return {"name": self.name, "arguments": json.dumps(self.arguments)}

    @staticmethod
    def from_openai_func_call(x: OpenAIFunctionCall | OpenAIFunction) -> "FunctionCall":
        return FunctionCall(
            name=x.name,
            arguments=json.loads(x.arguments),
        )


@dataclass
class ToolCall:
    id: str
    function: FunctionCall
    type: Literal["function"]

    def to_openai_tool_call(self) -> ChatCompletionMessageToolCallParam:
        return {
            "id": self.id,
            "function": self.function.to_openai_func_call_dict(),
            "type": self.type,
        }


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"

    @staticmethod
    def from_str(s: str) -> "Role":
        assert s in ["system", "user", "assistant", "function"]
        return Role(s)


class ContentPartText:
    def __init__(self, content: str) -> None:
        self.content = content

    def to_openai_content_part(self) -> ChatCompletionContentPartTextParam:
        return {"type": "text", "text": self.content}


class ContentPartImage:
    def __init__(self, url: str) -> None:
        self.url = url

    def to_openai_content_part(self) -> ChatCompletionContentPartImageParam:
        return {"type": "image_url", "image_url": {"url": self.url}}


ContentPart = Union[ContentPartText, ContentPartImage]


@dataclass
class Message:
    Role = Role
    role: Role
    """The role of the messages author.

    One of `system`, `user`, `assistant`, or `function`.
    """
    content: Optional[str | Sequence[ContentPart]] = None
    """The contents of the message.

    `content` is required for all messages, and may be null for assistant messages
    with function calls.
    """
    name: Optional[str] = None
    """
    Used for function messages to indicate the name of the function that was called.
    Function return data is provided in the `content` field.
    """
    function_call: Optional[FunctionCall] = None
    """
    DEPRECATED. Use `tool_calls` instead.
    The name and arguments of a function that should be called, as generated by the
    model.
    """
    tool_calls: Sequence[ToolCall] = field(default_factory=list)
    """The tool calls generated by the model, such as function calls."""
    tool_call_id: Optional[str] = None
    """Tool call that this message is responding to."""

    # def __post_init__(self):
    # if self.function_call is not None and isinstance(self.function_call.arguments, str):
    #     self.function_call = json.loads(json.dumps((self.function_call)))

    def to_json(self) -> JSON:
        data: Mapping[str, Any] = {
            "role": self.role,
            "content": self.content,
        }
        if self.name is not None:
            data["name"] = self.name
        if self.function_call is not None:
            data["function_call"] = {
                "name": self.function_call.name,
                "arguments": self.function_call.arguments,
            }
        return data

    def to_chat_completion_message_param(self) -> ChatCompletionMessageParam:
        content = self.content or ""
        if self.role == Role.SYSTEM:
            assert isinstance(content, str)
            return ChatCompletionSystemMessageParam(role="system", content=content)
        if self.role == Role.FUNCTION:
            assert isinstance(content, str)
            assert self.name is not None
            return ChatCompletionFunctionMessageParam(
                role="function", name=self.name, content=content
            )
        if self.role == Role.TOOL:
            assert isinstance(content, str)
            assert self.tool_call_id is not None
            return ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id=self.tool_call_id,
                content=content,
            )
        if self.role == Role.USER:
            _content = (
                content
                if isinstance(content, str)
                else [c.to_openai_content_part() for c in content]
            )
            return ChatCompletionUserMessageParam(role="user", content=_content)
        if self.role == Role.ASSISTANT:
            assert isinstance(content, str)
            if self.function_call is not None:
                return ChatCompletionAssistantMessageParam(
                    role="assistant",
                    content=content,
                    function_call=self.function_call.to_openai_func_call_dict(),
                )
            if len(self.tool_calls) > 0:
                return ChatCompletionAssistantMessageParam(
                    role="assistant",
                    content=content,
                    tool_calls=[
                        tool_call.to_openai_tool_call() for tool_call in self.tool_calls
                    ],
                )
            return ChatCompletionAssistantMessageParam(
                role="assistant", content=content
            )
        raise RuntimeError("Unreachable")

    @staticmethod
    def from_chat_completion_message(m: ChatCompletionMessage) -> "Message":
        return Message(
            role=Role.from_str(m.role),
            content=m.content,
            name=m.function_call.name if m.function_call is not None else None,
            function_call=(
                FunctionCall.from_openai_func_call(m.function_call)
                if m.function_call is not None
                else None
            ),
            tool_calls=(
                [
                    ToolCall(
                        id=t.id,
                        function=FunctionCall.from_openai_func_call(t.function),
                        type=t.type,
                    )
                    for t in m.tool_calls
                ]
                if m.tool_calls is not None
                else []
            ),
        )

    async def __aiter__(self):
        if self.content is not None:
            yield self.content

    async def message(self) -> "Message":
        return self


@dataclass
class ServerError(RuntimeError):
    message: Optional[str] = None


class MessageStream:
    def __aiter__(self) -> AsyncIterator[str]:
        raise NotImplementedError()

    async def message(self) -> Message:
        raise NotImplementedError()


class ChatMessageStream(MessageStream):
    def __init__(
        self,
        response: openai.AsyncStream[ChatCompletionChunk],
    ):
        self.__response = response
        self.__aiter = response.__aiter__()
        self.__message = Message(role=Role.ASSISTANT)
        self.__tool_calls: list[ChoiceDeltaToolCall] = []
        self.__final_message: Optional[Message] = None

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
            if d.index < len(self.__tool_calls):
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
                assert d.index == len(self.__tool_calls)
                assert d.function is not None
                self.__tool_calls.append(d)

    async def __anext__impl(self) -> str:
        if self.__final_message is not None:
            raise StopAsyncIteration()
        try:
            chunk = await self.__aiter.__anext__()
        except StopAsyncIteration:
            if self.__message.function_call is not None:
                args = cast(str, self.__message.function_call.arguments).strip()
                if len(args) == 0:
                    self.__message.function_call.arguments = {}
                else:
                    self.__message.function_call.arguments = json.loads(args)
            self.__message.tool_calls = self.__get_final_merged_tool_calls()
            self.__final_message = self.__message
            raise StopAsyncIteration()
        if hasattr(chunk, "error"):
            raise ServerError(chunk.error["message"])  # type: ignore
        delta = chunk.choices[0].delta
        # merge self.__message and delta
        if delta.content is not None:
            if self.__message.content is None:
                self.__message.content = ""
            assert isinstance(delta.content, str)
            assert isinstance(self.__message.content, str)
            self.__message.content += delta.content
        if delta.function_call is not None:
            if self.__message.function_call is None:
                self.__message.function_call = FunctionCall(name="", arguments="")
            if delta.function_call.name is not None:
                self.__message.function_call.name += delta.function_call.name
            if delta.function_call.arguments is not None:
                s = cast(str, self.__message.function_call.arguments or "")
                self.__message.function_call.arguments = (
                    s + delta.function_call.arguments
                )
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

    async def message(self) -> Message:
        if self.__final_message is not None:
            return self.__final_message
        async for _ in self:
            ...
        assert self.__final_message is not None
        return self.__final_message
