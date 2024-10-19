from dataclasses import dataclass
from enum import Enum, StrEnum
import inspect
from inspect import Parameter
import json
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TYPE_CHECKING,
    get_args,
    get_origin,
)

from agentia.agent import ToolCallEvent
from . import LOGGER, MSG_LOGGER

from .message import JSON, FunctionCall, Message, Role, ToolCall

from .plugins import Plugin

Tool = Plugin | Callable[..., Any]

Tools = Sequence[Tool]

if TYPE_CHECKING:
    from .agent import Agent

NAME_TAG = "agentia_tool_name"
DISPLAY_NAME_TAG = "agentia_tool_display_name"
IS_TOOL_TAG = "agentia_tool_is_tool"


@dataclass
class ToolInfo:
    name: str
    display_name: str
    description: str
    parameters: dict[str, Any]
    callable: Callable[..., Any]

    def to_json(self) -> JSON:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolRegistry:
    def __init__(self, agent: "Agent", tools: Tools | None = None) -> None:
        self.__functions: Dict[str, ToolInfo] = {}
        self.__plugins: Any = {}
        self.__agent = agent
        for t in tools or []:
            if inspect.isfunction(t):
                self.__add_function(t)
            elif isinstance(t, Plugin):
                self.__add_plugin(t)
        names = ", ".join([f"{k}" for k in self.__functions.keys()])
        LOGGER.debug(f"Registered Tools: {names}")

    def __add_function(self, f: Callable[..., Any]):
        fname = getattr(f, NAME_TAG, f.__name__)
        params: Any = {"type": "object", "properties": {}, "required": []}
        for pname, param in inspect.signature(f).parameters.items():
            # Skip self parameter
            if pname == "self":
                continue
            # Get parameter info
            prop = {}
            # Get parameter type inside Annotated
            t = param.annotation
            t = t if get_origin(t) != Annotated else get_args(t)[0]
            if t == inspect.Parameter.empty:
                t = str  # the default type is string
            # Get parameter optionality
            param_t_is_opt = False
            if get_origin(t) == Optional:
                param_t, param_t_is_opt = get_args(t)[0], True
            param_default_is_empty = param.default == inspect.Parameter.empty
            required = not param_t_is_opt and param_default_is_empty
            # Get parameter type
            assert get_origin(t) != Optional
            match t:
                # string type
                case x if x == str:
                    prop["type"] = "string"
                # integer type
                case x if x == int:
                    prop["type"] = "integer"
                # string enum
                case x if issubclass(x, StrEnum) or issubclass(x, Enum):
                    prop["type"] = "string"
                    for arg in x:
                        if not isinstance(arg, str):
                            raise ValueError(
                                f"{fname}.{pname}: Enum members must be strings only"
                            )
                    prop["enum"] = [x.value for x in x]
                case x if get_origin(x) == Literal:
                    prop["type"] = "string"
                    args = get_args(x)
                    for arg in args:
                        if not isinstance(arg, str):
                            raise ValueError(
                                f"{fname}.{pname}: Literal members must be strings only"
                            )
                    prop["enum"] = [x for x in args]
                case _other:
                    assert (
                        False
                    ), f"Invalid type annotation for parameter `{pname}` in function {fname}"
            # Get parameter description
            annotated_meta = (
                get_args(param.annotation)[1]
                if get_origin(param.annotation) == Annotated
                else None
            )
            if desc := annotated_meta if isinstance(annotated_meta, str) else None:
                prop["description"] = desc
            # Add non-optional parameter to the required list
            if required:
                params["required"].append(pname)
            # Add the parameter to the properties
            params["properties"][pname] = prop

        tool_info = ToolInfo(
            name=fname,
            display_name=getattr(f, DISPLAY_NAME_TAG, fname),
            description=f.__doc__ or "",
            parameters=params,
            callable=f,
        )
        self.__functions[tool_info.name] = tool_info
        return tool_info

    def __add_plugin(self, p: Plugin):
        # Add all functions from the plugin
        for _n, method in inspect.getmembers(p, predicate=inspect.ismethod):
            if not getattr(method, IS_TOOL_TAG, False):
                continue
            clsname = p.__class__.__name__
            if clsname.endswith("Plugin"):
                clsname = clsname[:-6]
            tool_info = self.__add_function(method)
            if not tool_info.name.startswith(clsname + "__"):
                old_name = tool_info.name
                tool_info.name = clsname + "__" + tool_info.name
                del self.__functions[old_name]
                self.__functions[tool_info.name] = tool_info
        # Add the plugin to the list of plugins
        clsname = p.__class__.__name__
        if clsname.endswith("Plugin"):
            clsname = clsname[:-6]
        self.__plugins[clsname] = p
        # Call the plugin's register method
        p.register(self.__agent)

    def get_plugin(self, name: str) -> Plugin:
        return self.__plugins[name]

    def is_empty(self) -> bool:
        return len(self.__functions) == 0

    def to_json(self) -> list[JSON]:
        functions = [v.to_json() for (k, v) in self.__functions.items()]
        return [{"type": "function", "function": f} for f in functions]

    def to_gpts_json(self, url: str) -> Any:
        while url.endswith("/"):
            url = url[:-1]
        functions: list[Any] = [v.to_json() for (k, v) in self.__functions.items()]
        return {
            "openapi": "3.1.0",
            "info": {
                "title": self.__agent.name or "",
                "description": self.__agent.description or "",
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
                for x in functions
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
                    if p.name == "__context__" and p.name not in args:
                        # kw_args[p.name] = context
                        raise ValueError(f"__context__ is not supported")
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
        ...
        # info = ToolInfo.from_fn(func)
        # if self.__client.on_tool_start is not None and tool_id is not None:
        #     await self.__client.on_tool_start(tool_id, info, args)

    async def __on_tool_end(
        self,
        func: Callable[..., Any],
        tool_id: str | None,
        args: JSON,
        result: JSON,
    ):
        ...
        # info = ToolInfo.from_fn(func)
        # if self.__client.on_tool_end is not None and tool_id is not None:
        #     await self.__client.on_tool_end(tool_id, info, args, result)

    async def call_function_raw(
        self, name: str, args: JSON, tool_id: str | None
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
            return {"error": f"Tool `{name}` not found"}
        func = self.__functions[name].callable
        raw_args = args
        args, kw_args = self.__filter_args(func, args)
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
        self, function_call: FunctionCall, tool_id: Optional[str]
    ) -> Any:
        func_name = function_call.name
        arguments = function_call.arguments
        result = await self.call_function_raw(func_name, arguments, tool_id)
        return result

    async def call_tools(self, tool_calls: Sequence[ToolCall]):
        for t in tool_calls:
            assert t.type == "function"
            yield ToolCallEvent(id=t.id, function=t.function)
            raw_result = await self.call_function(t.function, tool_id=t.id)
            yield ToolCallEvent(id=t.id, function=t.function, result=raw_result)
            if not isinstance(raw_result, str):
                result = json.dumps(raw_result)
            else:
                result = raw_result
            if t.id is not None:
                result_msg = Message(role="tool", tool_call_id=t.id, content=result)
            else:
                raise NotImplementedError("legacy functions not supported")
            yield result_msg
            await self.on_new_chat_message(result_msg)

    async def on_new_chat_message(self, msg: Message):
        for p in self.__plugins.values():
            result = p.on_new_chat_message(msg)
            if inspect.iscoroutine(result):
                await result
