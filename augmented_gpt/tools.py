from dataclasses import dataclass
import inspect
from inspect import Parameter
import json
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING
from . import LOGGER, MSG_LOGGER

from .message import JSON, FunctionCall, Message, Role, ToolCall

from .plugins import Plugin

Tool = Plugin | Callable[..., Any]

Tools = Sequence[Tool]

if TYPE_CHECKING:
    from .augmented_gpt import AugmentedGPT

TOOL_INFO_TAG = "gpt_function_call_info"


@dataclass
class ToolInfo:
    name: str
    display_name: str
    description: str
    parameters: dict[str, Any]

    @staticmethod
    def from_fn_opt(f: Callable[..., Any]) -> Optional["ToolInfo"]:
        if not hasattr(f, TOOL_INFO_TAG):
            return None
        info = getattr(f, TOOL_INFO_TAG)
        assert isinstance(info, ToolInfo) or info is None
        return info

    @staticmethod
    def from_fn(f: Callable[..., Any]) -> "ToolInfo":
        info = ToolInfo.from_fn_opt(f)
        assert info is not None
        return info

    def to_json(self) -> JSON:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolRegistry:
    def __init__(self, client: "AugmentedGPT", tools: Tools | None = None) -> None:
        self.__functions: Dict[str, Tuple[Any, Callable[..., Any]]] = {}
        self.__plugins: Any = {}
        self.__client = client
        for t in tools or []:
            if inspect.isfunction(t):
                self.__add_function(t)
            elif isinstance(t, Plugin):
                self.__add_plugin(t)
        names = ", ".join([f"{k}" for k in self.__functions.keys()])
        LOGGER.debug(f"Registered Tools: {names}")

    def __add_function(self, f: Callable[..., Any]):
        func_info = ToolInfo.from_fn(f)
        self.__functions[func_info.name] = (func_info.to_json(), f)

    def __add_plugin(self, p: Plugin):
        # Add all functions from the plugin
        for _n, method in inspect.getmembers(p, predicate=inspect.ismethod):
            func_info = ToolInfo.from_fn_opt(method)
            if func_info is None:
                continue
            clsname = p.__class__.__name__
            if clsname.endswith("Plugin"):
                clsname = clsname[:-6]
            if not func_info.name.startswith(clsname + "__"):
                func_info.name = clsname + "__" + func_info.name
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

    def to_gpts_json(self, url: str) -> Any:
        while url.endswith("/"):
            url = url[:-1]
        return {
            "openapi": "3.1.0",
            "info": {
                "title": self.__client.name or "",
                "description": self.__client.description or "",
                "version": "v1.0.0",
            },
            "servers": [{"url": url}],
            "paths": {
                f"/actions/{x['name']}": {
                    "get": {
                        "description": x["description"],
                        "operationId": x["name"],
                        "parameters": [
                            {
                                "name": name,
                                "in": "query",
                                "description": p["description"],
                                "required": name in x["parameters"]["required"],
                                "schema": {"type": p["type"]},
                            }
                            for name, p in x["parameters"]["properties"].items()
                        ],
                        "deprecated": False,
                    }
                }
                for (x, _) in self.__functions.values()
            },
            "components": {"schemas": {}},
        }

    def __filter_args(self, callable: Callable[..., Any], args: Any, context: Any):
        p_args: List[Any] = []
        kw_args: Dict[str, Any] = {}
        for p in inspect.signature(callable).parameters.values():
            match p.kind:
                case Parameter.POSITIONAL_ONLY:
                    p_args.append(args[p.name] if p.name in args else p.default.default)
                case Parameter.POSITIONAL_OR_KEYWORD | Parameter.KEYWORD_ONLY:
                    if p.name == "__context__" and p.name not in args:
                        kw_args[p.name] = context
                    else:
                        kw_args[p.name] = (
                            args[p.name] if p.name in args else p.default.default
                        )
                case other:
                    raise ValueError(f"{other} is not supported")
        return p_args, kw_args

    async def __on_tool_start(
        self, func: Callable[..., Any], tool_id: str | None, args: JSON
    ):
        info = ToolInfo.from_fn(func)
        if self.__client.on_tool_start is not None and tool_id is not None:
            await self.__client.on_tool_start(tool_id, info, args)

    async def __on_tool_end(
        self,
        func: Callable[..., Any],
        tool_id: str | None,
        args: JSON,
        result: JSON,
    ):
        info = ToolInfo.from_fn(func)
        if self.__client.on_tool_end is not None and tool_id is not None:
            await self.__client.on_tool_end(tool_id, info, args, result)

    async def call_function_raw(
        self, name: str, args: JSON, tool_id: str | None, context: Any = None
    ) -> Any:
        args_s = ""
        if isinstance(args, dict):
            for k, v in args.items():
                args_s += f"{k}={v}, "
            args_s = args_s[:-2]
        else:
            args_s = str(args)
        if tool_id is not None:
            MSG_LOGGER.info(f"GPT-Tool[{tool_id}] {name} {args_s}")
        else:
            MSG_LOGGER.info(f"GPT-Function {name} {args_s}")
        # key = func_name if not func_name.startswith("functions.") else func_name[10:]
        if name not in self.__functions:
            return {"error": f"Function or tool `{name}` not found"}
        func = self.__functions[name][1]
        raw_args = args
        args, kw_args = self.__filter_args(func, args, context)
        await self.__on_tool_start(func, tool_id, raw_args)
        try:
            result_or_coroutine = func(*args, **kw_args)
            if inspect.iscoroutine(result_or_coroutine):
                result = await result_or_coroutine
            else:
                result = result_or_coroutine
        except BaseException as e:
            MSG_LOGGER.error(f"Failed to run tool `{name}`: {e}")
            result = {"error": f"Failed to run tool `{name}`: {e}"}
        await self.__on_tool_end(func, tool_id, raw_args, result)
        result_s = json.dumps(result)
        if tool_id is not None:
            MSG_LOGGER.info(f"GPT-Tool[{tool_id}] {name} -> {result_s}")
        else:
            MSG_LOGGER.info(f"GPT-Function {name} -> {result_s}")
        return result

    async def call_function(
        self, function_call: FunctionCall, tool_id: Optional[str], context: Any
    ) -> Message:
        func_name = function_call.name
        arguments = function_call.arguments
        result = await self.call_function_raw(
            func_name, arguments, tool_id, context=context
        )
        if not isinstance(result, str):
            result = json.dumps(result)
        if tool_id is not None:
            result_msg = Message(role=Role.TOOL, tool_call_id=tool_id, content=result)
        else:
            result_msg = Message(role=Role.FUNCTION, name=func_name, content=result)
        return result_msg

    async def call_tools(
        self, tool_calls: Sequence[ToolCall], context: Any
    ) -> list[Message]:
        results: list[Message] = []
        for t in tool_calls:
            assert t.type == "function"
            result = await self.call_function(t.function, tool_id=t.id, context=context)
            results.append(result)
            await self.on_new_chat_message(result)
        return results

    async def on_new_chat_message(self, msg: Message):
        for p in self.__plugins.values():
            result = p.on_new_chat_message(msg)
            if inspect.iscoroutine(result):
                await result
