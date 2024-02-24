import asyncio
from typing import (
    AsyncGenerator,
    Literal,
    Any,
    overload,
)

from augmented_gpt.augmented_gpt import ChatCompletion
from augmented_gpt.gpt import LLMBackend, GPTModel, GPTOptions
from augmented_gpt.tools import ToolRegistry

from ..message import *

import openai
from openai._types import NOT_GIVEN
from openai.types.beta import Thread as OpenAIThread, Assistant as OpenAIAssistant
from openai.types.beta.threads.run import Run as OpenAIRun
from openai.types.beta.threads.run_submit_tool_outputs_params import ToolOutput
from augmented_gpt.message import FunctionCall, Message


class GPTAssistantBackend(LLMBackend):
    __assistant: OpenAIAssistant

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
        self.__instructions = instructions
        self.__assistant = self.__create_or_retrieve_assistant(None)
        self.__thread = self.__create_thread_sync(None)

    @property
    def id(self):
        return self.__assistant.id

    def __create_or_retrieve_assistant(self, id: Optional[str]):
        client = openai.OpenAI(api_key=self.api_key)
        tools: list[Any] = [{"type": "code_interpreter"}]
        for t in self.tools.to_json():
            tools.append(t)
        if id is None:
            return client.beta.assistants.create(
                name=self.name or NOT_GIVEN,
                description=self.description or NOT_GIVEN,
                instructions=self.__instructions or NOT_GIVEN,
                tools=tools,
                model=self.model,
            )
        else:
            asst = client.beta.assistants.retrieve(id)
            asst = client.beta.assistants.update(
                id,
                name=self.name or NOT_GIVEN,
                description=self.description or NOT_GIVEN,
                instructions=self.__instructions or NOT_GIVEN,
                tools=tools,
            )
            print(asst)
            return asst

    def __create_thread_sync(self, id: Optional[str]):
        client = openai.OpenAI(api_key=self.api_key)
        return Thread(self, client.beta.threads.create())

    async def create_thread(self):
        return Thread(self, await self.client.beta.threads.create())

    async def retrieve_thread(self, thread_id: str):
        return Thread(self, await self.client.beta.threads.retrieve(thread_id))

    async def delete_thread(self, thread_id: str):
        await self.client.beta.threads.delete(thread_id)

    @overload
    async def __chat_completion(
        self, messages: list[Message], stream: Literal[False] = False
    ) -> AsyncGenerator[Message, None]: ...

    @overload
    async def __chat_completion(
        self, messages: list[Message], stream: Literal[True]
    ) -> AsyncGenerator[MessageStream, None]: ...

    async def __chat_completion(  # type: ignore
        self, messages: list[Message], stream: bool = False
    ):
        assert not stream
        # Add messages
        for m in messages:
            await self._on_new_chat_message(m)
            assert isinstance(m.content, str)
            await self.__thread.add(m.content)
        # # Run the thread
        run = await self.__thread.run()
        async for m in run:
            msg = Message(role=Role.USER, content=m)
            yield msg

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


class Thread:
    def __init__(self, assistant: GPTAssistantBackend, thread: OpenAIThread):
        self.assistant = assistant
        self.__thread = thread

    @property
    def id(self) -> str:
        return self.__thread.id

    async def reset(self):
        await self.assistant.delete_thread(self.id)
        new_thread = await self.assistant.create_thread()
        self.__thread = new_thread.__thread

    async def update(self):
        self.__thread = await self.assistant.client.beta.threads.retrieve(self.id)

    async def add(self, content: str) -> str:
        msg = await self.assistant.client.beta.threads.messages.create(
            self.__thread.id, content=content, role="user"
        )
        return msg.id

    async def run(self):
        latest_msg_id: str | None = None
        messages = await self.assistant.client.beta.threads.messages.list(
            thread_id=self.id
        )
        async for m in messages:
            latest_msg_id = m.id
            break
        run = await self.assistant.client.beta.threads.runs.create(
            thread_id=self.__thread.id,
            assistant_id=self.assistant.id,
        )
        return Run(self, run, latest_msg_id)

    async def send(self, content: str):
        msg = await self.assistant.client.beta.threads.messages.create(
            self.__thread.id, content=content, role="user"
        )
        run = await self.assistant.client.beta.threads.runs.create(
            thread_id=self.__thread.id,
            assistant_id=self.assistant.id,
        )
        return Run(self, run, msg.id)


class Run:
    def __init__(self, thread: Thread, run: OpenAIRun, latest_msg_id: str | None):
        self.thread = thread
        self.__run = run
        self.__latest_msg_id = latest_msg_id

    @property
    def id(self):
        return self.__run.id

    async def __do_actions(self):
        assert self.__run.required_action is not None
        tool_calls = self.__run.required_action.submit_tool_outputs.tool_calls
        print(tool_calls)
        results = await self.thread.assistant.tools.call_tools(
            [
                ToolCall(
                    type="function",
                    function=FunctionCall.from_openai_func_call(t.function),
                    id=t.id,
                )
                for t in tool_calls
            ]
        )
        tool_outputs: list[ToolOutput] = []
        for m in results:
            assert isinstance(m, Message)
            assert isinstance(m.content, str)
            assert m.tool_call_id is not None
            tool_outputs.append(
                ToolOutput(output=m.content, tool_call_id=m.tool_call_id)
            )
        await self.thread.assistant.client.beta.threads.runs.submit_tool_outputs(
            run_id=self.id,
            thread_id=self.thread.id,
            tool_outputs=tool_outputs,
        )

    async def __get_new_messages(self) -> list[str]:
        messages = await self.thread.assistant.client.beta.threads.messages.list(
            thread_id=self.thread.id
        )
        new_messages: list[str] = []
        first_mid = None
        async for m in messages:
            if m.id == self.__latest_msg_id:
                break
            for c in m.content:
                if c.type == "text":
                    new_messages.append(c.text.value)
            if first_mid is None:
                first_mid = m.id
        if first_mid is not None:
            self.__latest_msg_id = first_mid
        return new_messages

    async def __aiter__(self):
        while True:
            self.__run = await self.thread.assistant.client.beta.threads.runs.retrieve(
                run_id=self.__run.id,
                thread_id=self.thread.id,
            )
            for m in await self.__get_new_messages():
                yield m
            match self.__run.status:
                case "completed":
                    return
                case "queued" | "in_progress":
                    await asyncio.sleep(1)
                    continue
                case "cancelled" | "cancelling" | "failed" | "expired":
                    for m in await self.__get_new_messages():
                        yield m
                    raise RuntimeError(self.__run.status)
                case "requires_action":
                    await self.__do_actions()