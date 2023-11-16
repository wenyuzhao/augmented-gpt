import asyncio
import json
import os
from typing import Any, Awaitable, Optional

import openai
from openai._types import NOT_GIVEN
from openai.types.beta import Thread as OpenAIThread
from openai.types.beta.threads.run import Run as OpenAIRun
from openai.types.beta.threads.run_submit_tool_outputs_params import ToolOutput
from augmented_gpt.message import FunctionCall, Message

from .tools import ToolRegistry, Tools


class AugmentedGPTAssistant:
    def __init__(
        self,
        api_key: Optional[str] = None,
        id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        instructions: Optional[str] = None,
        tools: Optional[Tools] = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        assert self.api_key is not None, "Missing OPENAI_API_KEY"
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.tools = ToolRegistry(self.client, tools)
        self.__assistant = self.__create_or_retrieve_assistant(
            id, name, description, instructions
        )

    @property
    def id(self):
        return self.__assistant.id

    def __create_or_retrieve_assistant(
        self,
        id: Optional[str],
        name: Optional[str],
        description: Optional[str],
        instructions: Optional[str],
    ):
        client = openai.OpenAI(api_key=self.api_key)
        tools: list[Any] = [{"type": "code_interpreter"}]
        for x in self.tools.to_json():
            print(x)
            tools.append(x)
        if id is None:
            return client.beta.assistants.create(
                name=name or NOT_GIVEN,
                description=description or NOT_GIVEN,
                instructions=instructions or NOT_GIVEN,
                tools=tools,
                model="gpt-4-1106-preview",
            )
        else:
            asst = client.beta.assistants.retrieve(id)
            asst = client.beta.assistants.update(
                id,
                name=name or NOT_GIVEN,
                description=description or NOT_GIVEN,
                instructions=instructions or NOT_GIVEN,
                tools=tools,
            )
            print(asst)
            return asst

    async def create_thread(self):
        return Thread(self, await self.client.beta.threads.create())

    async def retrieve_thread(self, thread_id: str):
        return Thread(self, await self.client.beta.threads.retrieve(thread_id))

    async def delete_thread(self, thread_id: str):
        await self.client.beta.threads.delete(thread_id)


class Thread:
    def __init__(self, assistant: AugmentedGPTAssistant, thread: OpenAIThread):
        self.assistant = assistant
        self.__thread = thread

    @property
    def id(self):
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
        tasks: list[Awaitable[Message]] = []
        for t in tool_calls:
            assert t.type == "function"
            func = FunctionCall(
                name=t.function.name, arguments=json.loads(t.function.arguments)
            )
            tasks.append(self.thread.assistant.tools.call_function(func, tool_id=t.id))
        msgs = await asyncio.gather(*tasks)
        tool_outputs: list[ToolOutput] = []
        for m in msgs:
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
                    await asyncio.sleep(3)
                    continue
                case "cancelled" | "cancelling" | "failed" | "expired":
                    for m in await self.__get_new_messages():
                        yield m
                    raise RuntimeError(self.__run.status)
                case "requires_action":
                    await self.__do_actions()
