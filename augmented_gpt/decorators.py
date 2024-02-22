from typing import (
    Callable,
    Coroutine,
    List,
    TypeVar,
    Any,
    Optional,
)
import inspect

from augmented_gpt.message import JSON


class _Param:
    def __init__(
        self,
        description: Optional[str] = None,
        default: Optional[Any] = None,
        enum: Optional[List[Any]] = None,
    ):
        self.description = description
        self.default = default
        self.enum = enum


def param(
    description: Optional[str] = None,
    default: Optional[Any] = None,
    enum: Optional[List[Any]] = None,
) -> Any:
    return _Param(description, default, enum)


R = TypeVar("R", Coroutine[Any, Any, Optional[JSON | str]], Optional[JSON | str])


def tool(callable: Callable[..., R]) -> Callable[..., R]:
    # Get function name
    fname = callable.__name__
    # Get parameter info
    params: Any = {
        "type": "object",
        "properties": {},
        "required": [],
    }
    for pname, param in inspect.signature(callable).parameters.items():
        if pname == "self":
            continue
        assert isinstance(
            param.default, _Param
        ), f"Invalid default value for parameter `{pname}` in function {fname}"
        prop = {}
        required = True
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
        if required:
            params["required"].append(pname)
        else:
            assert (
                param.default.default is not None
            ), f"Optional parameter `{pname}` in function {fname} requires a default value"
        params["properties"][pname] = prop
    # store gpt function metadata to the callable object
    setattr(
        callable,
        "gpt_function_call_info",
        {
            "name": fname,
            "description": callable.__doc__ or "",
            "parameters": params,
        },
    )
    return callable
