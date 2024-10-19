from typing import (
    Callable,
    Coroutine,
    TypeVar,
    Any,
    Optional,
    overload,
)
import inspect

from agentia.message import JSON


class _Param:
    def __init__(
        self,
        description: Optional[str] = None,
        default: Optional[str | int] = None,
        enum: Optional[list[str] | list[int]] = None,
    ):
        self.description = description
        self.default = default
        self.enum = enum


def param(
    description: Optional[str] = None,
    default: Optional[str | int] = None,
    enum: Optional[list[str] | list[int]] = None,
) -> Any:
    return _Param(description, default, enum)


R = TypeVar("R", Coroutine[Any, Any, Optional[JSON | str]], Optional[JSON | str])


@overload
def tool(name: Callable[..., R]) -> Callable[..., R]: ...
@overload
def tool(
    name: str | None = None, display_name: str | None = None
) -> Callable[..., Callable[..., R]]: ...


def tool(
    name: str | Callable[..., R] | None = None, display_name: Optional[str] = None
) -> Callable[..., R] | Callable[[Callable[..., R]], Callable[..., R]]:

    def __tool_impl(callable: Callable[..., R]) -> Callable[..., R]:
        # Get function name
        fname = callable.__name__ if not isinstance(name, str) else name
        # Get parameter info
        params: Any = {"type": "object", "properties": {}, "required": []}
        for pname, param in inspect.signature(callable).parameters.items():
            if pname == "self":
                continue
            assert (
                isinstance(param.default, _Param) or param.name == "__context__"
            ), f"Invalid default value for parameter `{pname}` in function {fname}"
            if not isinstance(param.default, _Param):
                continue
            prop, required = {}, True
            match param.annotation:
                case x if x == str or x == Optional[str]:
                    required = x == str
                    prop["type"] = "string"
                case x if x == int or x == Optional[int]:
                    required = x == int
                    prop["type"] = "integer"
                case _other:
                    assert (
                        False
                    ), f"Invalid type annotation for parameter `{pname}` in function {fname}"
            if param.default.description is not None:
                prop["description"] = param.default.description
            if param.default.enum is not None:
                prop["enum"] = param.default.enum
            if required:
                params["required"].append(pname)
            params["properties"][pname] = prop
        # store gpt function metadata to the callable object
        from agentia.tools import ToolInfo, TOOL_INFO_TAG

        setattr(
            callable,
            TOOL_INFO_TAG,
            ToolInfo(
                name=fname,
                display_name=display_name or fname,
                description=callable.__doc__ or "",
                parameters=params,
            ),
        )
        return callable

    if name is not None and (not isinstance(name, str)) and display_name is None:
        return __tool_impl(name)

    return __tool_impl
