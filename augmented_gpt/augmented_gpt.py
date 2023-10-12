from typing import Callable, Dict, Generator, List, Sequence, Tuple, Any, overload, TYPE_CHECKING

from .message import *
from dotenv import dotenv_values
import inspect
from inspect import Parameter
import openai
import logging
import os
from openai.types.chat import ChatCompletionMessageParam

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .plugins import Plugin

openai_api_key = None

def get_openai_api_key():
    global openai_api_key
    return openai_api_key

class AugmentedGPT:
    @staticmethod
    def set_api_key(key: str):
        global openai_api_key
        openai_api_key = key
    def __init__(
        self,
        model: str | Literal[
                "gpt-4",
                "gpt-4-0314",
                "gpt-4-0613",
                "gpt-4-32k",
                "gpt-4-32k-0314",
                "gpt-4-32k-0613",
                "gpt-3.5-turbo",
                "gpt-3.5-turbo-16k",
                "gpt-3.5-turbo-0301",
                "gpt-3.5-turbo-0613",
                "gpt-3.5-turbo-16k-0613",
            ] = "gpt-4",
        functions: List[Callable[..., Any]] = [],
        plugins: Sequence["Plugin"] = [],
        debug: bool = False,
    ):
        self.model = model
        api_key = dotenv_values().get(
            "OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", openai_api_key)
        )
        assert api_key is not None, "Missing OPENAI_API_KEY"
        openai.api_key = api_key
        self.logger = logging.getLogger("AugmentedGPT")
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)
        self.__functions: Dict[str, Tuple[Any, Callable[..., Any]]] = {}
        for f in functions:
            self.add_function(f)
        self.__plugins: Any = {}
        for p in plugins:
            clsname = p.__class__.__name__
            if clsname.endswith("Plugin"):
                clsname = clsname[:-6]
            p.register(self)
            self.__plugins[clsname] = p
        self.history: List[Message] = []

    def get_plugin(self, name: str) -> "Plugin":
        return self.__plugins[name]

    def add_function(self, f: Callable[..., Any]):
        func_info = getattr(f, "gpt_function_call_info")
        self.logger.info("Register-Function: " + func_info["name"])
        self.__functions[func_info["name"]] = (func_info, f)

    def __filter_args(self, callable: Callable[..., Any], args: Any):
        p_args: List[Any] = []
        kw_args: Dict[str, Any] = {}
        for p in inspect.signature(callable).parameters.values():
            match p.kind:
                case Parameter.POSITIONAL_ONLY:
                    p_args.append(args[p.name] if p.name in args else p.default.default)
                case Parameter.POSITIONAL_OR_KEYWORD | Parameter.KEYWORD_ONLY:
                    kw_args[p.name] = (
                        args[p.name] if p.name in args else p.default.default
                    )
                case other:
                    raise ValueError(f"{other} is not supported")
        return p_args, kw_args

    def __call_function(self, function_call: FunctionCall) -> Message:
        func_name = function_call.name
        func = self.__functions[func_name][1]
        arguments = function_call.arguments
        args, kw_args = self.__filter_args(func, arguments)
        result = func(*args, **kw_args)
        if not isinstance(result, str):
            result = json.dumps(result)
        return Message(role="function", name=func_name, content=result)

    @overload
    def __chat_completion_request(self, messages: List[Message], stream: Literal[False]) -> Message:
        ...

    @overload
    def __chat_completion_request(self, messages: List[Message], stream: Literal[True]) -> MessageStream:
        ...

    def __chat_completion_request(self, messages: List[Message], stream: bool) -> Message | MessageStream:
        functions = [x for (x, _) in self.__functions.values()]
        msgs: List[ChatCompletionMessageParam] = [m.to_chat_completion_message_param() for m in messages]
        if stream:
            response = openai.chat.completions.create(
                model=self.model,
                messages=msgs,
                functions=functions,
                function_call="auto",
                stream=True,
            )
            return MessageStream(response)
        else:
            response = openai.chat.completions.create(
                model=self.model,
                messages=msgs,
                functions=functions,
                function_call="auto",
            )
            return Message.from_chat_completion_message(response.choices[0].message)

    @overload
    def chat_completion(self, messages: List[Message], stream: Literal[False] = False, context_free: bool = False) -> Generator[Message, None, None]:
        ...

    @overload
    def chat_completion(self, messages: List[Message], stream: Literal[True], context_free: bool = False) -> Generator[MessageStream, None, None]:
        ...

    def chat_completion(
        self,
        messages: list[Message],
        stream: bool = False,
        context_free: bool = False,
    ):
        history = self.history if not context_free else []
        history.extend(messages)
        for m in messages:
            self.__on_new_chat_message(m)
        # First completion request
        message: Message
        if stream:
            s = self.__chat_completion_request(history, stream=True)
            yield s
            message = s.message()
        else:
            message = self.__chat_completion_request(history, stream=False)
            yield message
        history.append(message)
        self.__on_new_chat_message(message)
        while message.function_call is not None:
            # ChatGPT wanted to call a user-defined function
            result = self.__call_function(message.function_call)
            history.append(result)
            self.__on_new_chat_message(result)
            yield MessageStream(None, result) if stream else result
            # Send back the function call result
            message: Message
            if stream:
                r = self.__chat_completion_request(history, stream=True)
                yield r
                message = r.message()
            else:
                message = self.__chat_completion_request(history, stream=False)
                yield message
            history.append(message)
            self.__on_new_chat_message(message)

    def __on_new_chat_message(self, msg: Message):
        for p in self.__plugins.values():
            p.on_new_chat_message(msg)
