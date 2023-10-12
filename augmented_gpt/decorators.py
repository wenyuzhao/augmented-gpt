from typing import Callable, Dict, List, Union, Any, Optional, overload
import inspect


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


@overload
def function(description: Callable[..., Any]) -> Callable[..., Any]:
    ...


@overload
def function(
    description: Optional[str],
    name: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    ...


def function(
    description: Union[Optional[str], Callable[..., Any]] = None,
    name: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
):
    def decorator_func(callable: Callable[..., Any]) -> Callable[..., Any]:
        # Get function name
        fname = name if name is not None else callable.__name__
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
        setattr(
            callable,
            "gpt_function_call_info",
            {
                "name": fname,
                "description": description or callable.__doc__ or "",
                "parameters": params,
            },
        )
        return callable

    if inspect.isfunction(description) or inspect.ismethod(description):
        func = description
        description = None
        return decorator_func(func)
    return decorator_func
