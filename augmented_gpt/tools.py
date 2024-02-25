import inspect
from inspect import Parameter
import json
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

from augmented_gpt.message import JSON, FunctionCall, Message, Role, ToolCall

from .plugins import Plugin
from urllib.parse import urlparse

Tool = Plugin | Callable[..., Any]

Tools = Sequence[Tool]

if TYPE_CHECKING:
    from .augmented_gpt import AugmentedGPT


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

    def __add_function(self, f: Callable[..., Any]):
        func_info = getattr(f, "gpt_function_call_info")
        func_info = {
            "name": func_info["name"],
            "description": func_info["description"],
            "parameters": func_info["parameters"],
        }
        self.__functions[func_info["name"]] = (func_info, f)

    def __add_plugin(self, p: Plugin):
        # Add all functions from the plugin
        for _n, method in inspect.getmembers(p, predicate=inspect.ismethod):
            if not hasattr(method, "gpt_function_call_info"):
                continue
            func_info = getattr(method, "gpt_function_call_info")
            clsname = p.__class__.__name__
            if clsname.endswith("Plugin"):
                clsname = clsname[:-6]
            if not func_info["name"].startswith(clsname + "__"):
                func_info["name"] = clsname + "__" + func_info["name"]
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
        o = urlparse(url)
        host = o.scheme + "://" + o.netloc
        base_url = o.path
        return {
            "openapi": "3.1.0",
            "info": {
                "title": self.__client.name or "",
                "description": self.__client.description or "",
                "version": "v1.0.0",
            },
            "servers": [{"url": host}],
            "paths": {
                f"{base_url}/actions/{x['name']}": {
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

    async def __on_tool_start(
        self, func: Callable[..., Any], tool_id: str | None, args: JSON
    ):
        display_name = getattr(func, "gpt_function_call_info")["display_name"]
        if self.__client.on_tool_start is not None and tool_id is not None:
            await self.__client.on_tool_start(tool_id, display_name, args)

    async def __on_tool_end(
        self,
        func: Callable[..., Any],
        tool_id: str | None,
        args: JSON,
        result: JSON,
    ):
        display_name = getattr(func, "gpt_function_call_info")["display_name"]
        if self.__client.on_tool_end is not None and tool_id is not None:
            await self.__client.on_tool_end(tool_id, display_name, args, result)

    async def call_function_raw(
        self, name: str, args: JSON, tool_id: str | None
    ) -> Any:
        # key = func_name if not func_name.startswith("functions.") else func_name[10:]
        if name not in self.__functions:
            return {"error": f"Function or tool `{name}` not found"}
        func = self.__functions[name][1]
        raw_args = args
        args, kw_args = self.__filter_args(func, args)
        await self.__on_tool_start(func, tool_id, raw_args)
        try:
            result_or_coroutine = func(*args, **kw_args)
            if inspect.iscoroutine(result_or_coroutine):
                result = await result_or_coroutine
            else:
                result = result_or_coroutine
        except Exception as e:
            print(e)
            result = {"error": f"Failed to run tool `{name}`: {e}"}
        await self.__on_tool_end(func, tool_id, raw_args, result)
        return result

    async def call_function(
        self, function_call: FunctionCall, tool_id: Optional[str]
    ) -> Message:
        func_name = function_call.name
        arguments = function_call.arguments
        result = await self.call_function_raw(func_name, arguments, tool_id)
        if not isinstance(result, str):
            result = json.dumps(result)
        if tool_id is not None:
            result_msg = Message(role=Role.TOOL, tool_call_id=tool_id, content=result)
        else:
            result_msg = Message(role=Role.FUNCTION, name=func_name, content=result)
        return result_msg

    async def call_tools(self, tool_calls: Sequence[ToolCall]) -> list[Message]:
        results: list[Message] = []
        for t in tool_calls:
            assert t.type == "function"
            result = await self.call_function(t.function, tool_id=t.id)
            results.append(result)
            await self.on_new_chat_message(result)
        return results

    async def on_new_chat_message(self, msg: Message):
        for p in self.__plugins.values():
            result = p.on_new_chat_message(msg)
            if inspect.iscoroutine(result):
                await result
