import inspect
from inspect import Parameter
import json
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from openai import AsyncOpenAI

from augmented_gpt.message import JSON, FunctionCall, Message, Role

from .plugins import Plugin

Tool = Plugin | Callable[..., Any]

Tools = Sequence[Tool]


class ToolRegistry:
    def __init__(self, client: AsyncOpenAI, tools: Tools | None = None) -> None:
        self.__functions: Dict[str, Tuple[Any, Callable[..., Any]]] = {}
        self.__plugins: Any = {}
        self.__client = client
        for t in tools or []:
            if inspect.isfunction(t):
                self.__add_function(t)
            elif isinstance(t, Plugin):
                self.__add_plugin(t)

    def __add_function(self, f: Callable[..., Any]):
        func_info = getattr(f, "gpt_function_call_info")
        self.__functions[func_info["name"]] = (func_info, f)

    def __add_plugin(self, p: Plugin):
        # Add all functions from the plugin
        for _n, method in inspect.getmembers(p, predicate=inspect.ismethod):
            if not hasattr(method, "gpt_function_call_info"):
                continue
            func_info = getattr(method, "gpt_function_call_info")
            clsname = self.__class__.__name__
            if clsname.endswith("Plugin"):
                clsname = clsname[:-6]
            if not func_info["name"].startswith(clsname + "-"):
                func_info["name"] = clsname + "-" + func_info["name"]
            self.__add_function(method)
        # Add the plugin to the list of plugins
        clsname = p.__class__.__name__
        if clsname.endswith("Plugin"):
            clsname = clsname[:-6]
        self.__plugins[clsname] = p
        # Call the plugin's register method
        p.register(self.__client)

    def get_plugin(self, name: str) -> Plugin:
        return self.__plugins[name]

    def is_empty(self) -> bool:
        return len(self.__functions) == 0

    def to_json(self, legacy: bool = False) -> list[JSON]:
        functions = [x for (x, _) in self.__functions.values()]
        if legacy:
            return functions
        return [{"type": "function", "function": f} for f in functions]

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

    # async def call_tools(self, tool_calls: List[FunctionCall]) -> List[Message]:

    async def call_function(
        self, function_call: FunctionCall, tool_id: Optional[str]
    ) -> Message:
        func_name = function_call.name
        key = func_name if not func_name.startswith("functions.") else func_name[10:]
        if tool_id is not None:
            result_msg = Message(role=Role.TOOL, tool_call_id=tool_id, content="")
        else:
            result_msg = Message(role=Role.FUNCTION, name=func_name, content="")
        try:
            func = self.__functions[key][1]
            arguments = function_call.arguments
            args, kw_args = self.__filter_args(func, arguments)
            # self.logger.debug(
            #     f"➡️ {func_name}: "
            #     + ", ".join(str(a) for a in args)
            #     + ", ".join((f"{k}={v}" for k, v in kw_args.items()))
            # )
            result_or_coroutine = func(*args, **kw_args)
            if inspect.iscoroutine(result_or_coroutine):
                result = await result_or_coroutine
            else:
                result = result_or_coroutine
            if not isinstance(result, str):
                result = json.dumps(result)
            result_msg.content = result
        except Exception as e:
            print(e)
            result_msg.content = json.dumps(
                {"error": f"Failed to run tool `{func_name}`: {e}"}
            )
        return result_msg

    async def on_new_chat_message(self, msg: Message):
        for p in self.__plugins.values():
            result = p.on_new_chat_message(msg)
            if inspect.iscoroutine(result):
                await result
