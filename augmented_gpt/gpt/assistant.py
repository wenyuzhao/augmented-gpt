import asyncio
from pathlib import Path
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
        assistant_id: str | None = None,
        thread_id: str | None = None,
    ) -> None:
        super().__init__(
            model, tools, gpt_options, api_key, instructions, name, description, debug
        )
        self.__instructions = instructions
        self.__assistant = self.__create_or_retrieve_assistant(assistant_id)
        self.__thread = self.__create_or_reuse_thread_sync(thread_id)

    @property
    def id(self):
        return self.__assistant.id

    def reset(self):
        # Reset thread
        self.__thread.reset()
        # Delete files
        # self.delete_all_files()

    # def delete_all_files(self):
    #     client = openai.OpenAI(api_key=self.api_key)
    #     while True:
    #         files = client.beta.assistants.files.list(self.id, limit=100)
    #         if len(files.data) == 0:
    #             break
    #         for f in files.data:
    #             client.beta.assistants.files.delete(file_id=f.id, assistant_id=self.id)
    #             client.files.delete(file_id=f.id)

    def __create_or_retrieve_assistant(self, id: Optional[str]):
        client = openai.OpenAI(api_key=self.api_key)
        tools: list[Any] = [{"type": "code_interpreter"}, {"type": "file_search"}]
        for t in self.tools.to_json():
            tools.append(t)
        if id is None:
            ass = client.beta.assistants.create(
                name=self.name or NOT_GIVEN,
                description=self.description or NOT_GIVEN,
                instructions=self.__instructions or NOT_GIVEN,
                tools=tools,
                model=self.model,
            )
            print("Create assistant", ass.id)
            return ass
        else:
            ass = client.beta.assistants.retrieve(id)
            ass = client.beta.assistants.update(
                id,
                name=self.name or NOT_GIVEN,
                description=self.description or NOT_GIVEN,
                instructions=self.__instructions or NOT_GIVEN,
                tools=tools,
            )
            print("Reuse assistant", ass.id)
            return ass

    def __list_threads(self) -> list[str]:
        return []

    def __create_or_reuse_thread_sync(self, thread_id: str | None) -> "Thread":
        client = openai.OpenAI(api_key=self.api_key)
        # Reuse a user-provided thread
        if thread_id is not None:
            try:
                t = client.beta.threads.retrieve(thread_id)
                print("Reuse thread", t.id)
                return Thread(self, t)
            except BaseException as e:
                print(e)
                pass
        # If there is already a thread, retrieve and reuse it
        threads = self.__list_threads()
        if len(threads) > 0:
            t = client.beta.threads.retrieve(threads[0])
            print("Reuse thread", t.id)
        else:
            t = client.beta.threads.create()
            print("Create thread", t.id)
        return Thread(self, t)

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
            files = (
                [f if isinstance(f, Path) else Path(f) for f in m.files]
                if m.files is not None
                else None
            )
            await self.__thread.add(m.content, files=files)
        # # Run the thread
        run = await self.__thread.run()
        async for m in run:
            msg = Message(role=Role.ASSISTANT, content=m)
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

    def get_current_assistant_id(self) -> Optional[str]:
        return self.__assistant.id

    def get_current_thread_id(self) -> Optional[str]:
        return self.__thread.id

    def set_model(self, model: "GPTModel"):
        super().set_model(model)
        client = openai.OpenAI(api_key=self.api_key)
        client.beta.assistants.update(assistant_id=self.id, model=self.model)


class Thread:
    def __init__(self, assistant: GPTAssistantBackend, thread: OpenAIThread):
        self.assistant = assistant
        self.__thread = thread

    @property
    def id(self) -> str:
        return self.__thread.id

    def reset(self):
        client = openai.OpenAI(api_key=self.assistant.api_key)
        client.beta.threads.delete(self.id)
        new_thread = client.beta.threads.create()
        self.__thread = new_thread

    async def update(self):
        self.__thread = await self.assistant.client.beta.threads.retrieve(self.id)

    async def upload_file(self, file: Path) -> str:
        with open(file, "rb") as f:
            res = await self.assistant.client.files.create(file=f, purpose="assistants")
            print(f"Uploaded `{str(file)}` to OpenAI with id {res.id}.")
            return res.id

    async def add(self, content: str, files: list[Path] | None) -> str:
        file_ids = (
            [await self.upload_file(f) for f in files]
            if files is not None
            else NOT_GIVEN
        )
        msg = await self.assistant.client.beta.threads.messages.create(
            self.__thread.id,
            content=content,
            role="user",
            attachments=[{"file_id": fid, "add_to": ["file_search"]} for fid in file_ids or []],
        )
        return msg.id

    async def run(self):
        latest_msg_id: str | None = None
        messages = await self.assistant.client.beta.threads.messages.list(
            thread_id=self.id, order="desc", limit=1
        )
        for m in messages.data:
            latest_msg_id = m.id
            break
        run = await self.assistant.client.beta.threads.runs.create(
            thread_id=self.__thread.id,
            assistant_id=self.assistant.id,
        )
        return Run(self, run, latest_msg_id)


class Run:
    def __init__(self, thread: Thread, run: OpenAIRun, latest_msg_id: str | None):
        self.thread = thread
        self.__run = run
        self.__latest_msg_id: str | None = latest_msg_id

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
            thread_id=self.thread.id,
            after=self.__latest_msg_id or NOT_GIVEN,
            order="asc",
        )
        new_messages: list[str] = []
        for m in messages.data:
            for c in m.content:
                if c.type == "text":
                    new_messages.append(c.text.value)
            if len(m.content) > 0:
                self.__latest_msg_id = m.id
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
