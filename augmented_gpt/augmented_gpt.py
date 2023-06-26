from typing import *
from .message import *
from dotenv import dotenv_values
import inspect
from inspect import Parameter
import openai
import logging


class AugmentedGPT:
    def __init__(
        self,
        functions: List[Callable] = [],
        plugins: List["Plugin"] = [],
        debug=False,
    ):
        openai.api_key = dotenv_values()["OPENAI_API_KEY"]
        self.logger = logging.getLogger("AugmentedGPT")
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)
        self.__functions: Dict[str, Tuple[Any, Callable]] = {}
        for f in functions:
            self.add_function(f)
        self.__plugins = {}
        for p in plugins:
            clsname = p.__class__.__name__
            if clsname.endswith("Plugin"):
                clsname = clsname[:-6]
            p.register(self)
            self.__plugins[clsname] = p
        self.history: List[Message] = []

    def get_plugin(self, name: str) -> "Plugin":
        return self.__plugins[name]

    def add_function(self, f: Callable):
        func_info = getattr(f, "gpt_function_call_info")
        self.logger.info("Register-Function: " + func_info["name"])
        self.__functions[func_info["name"]] = (func_info, f)

    def __filter_args(self, callable: Callable, args: Any):
        p_args = []
        kw_args = {}
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

    def __call_function(self, function_call) -> Message:
        func_name = function_call["name"]
        func = self.__functions[func_name][1]
        arguments = json.loads(function_call["arguments"])
        args, kw_args = self.__filter_args(func, arguments)
        result = func(*args, **kw_args)
        if not isinstance(result, str):
            result = json.dumps(result)
        return Message(role="function", name=func_name, content=result)

    def __chat_completion_request(
        self, messages: List[Message], stream=False
    ) -> Union[Message, MessageStream]:
        functions = [x for (x, _) in self.__functions.values()]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=[m.to_json() for m in messages],
            functions=functions,
            function_call="auto",
            stream=stream,
        )
        if stream:
            return MessageStream(response)
        else:
            return Message(**response["choices"][0]["message"])

    def chat_completion(
        self,
        messages: Union[Message, List[Message]],
        stream=False,
        content_free=False,
    ):
        history = self.history if not content_free else []
        if isinstance(messages, list):
            history.extend(messages)
            for m in messages:
                self.__on_new_chat_message(m)
        else:
            history.append(messages)
            self.__on_new_chat_message(messages)
        response = self.__chat_completion_request(history, stream=stream)
        yield response
        message = response.message() if stream else response
        history.append(message)
        self.__on_new_chat_message(message)
        while message.function_call is not None:
            result = self.__call_function(message.function_call)
            history.append(result)
            self.__on_new_chat_message(result)
            yield MessageStream(None, result) if stream else result
            response = self.__chat_completion_request(history, stream=stream)
            yield response
            message = response.message() if stream else response
            history.append(message)
            self.__on_new_chat_message(message)

    def __on_new_chat_message(self, msg: Message):
        for p in self.__plugins.values():
            p.on_new_chat_message(msg)
