from typing import *
import json
from dataclasses import dataclass

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None


@dataclass
class Message:
    role: Literal["system", "user", "assistant", "function"]
    content: str
    name: Optional[str] = None
    function_call: Optional[JSON] = None

    def __post_init__(self):
        if self.function_call is not None:
            self.function_call = json.loads(json.dumps((self.function_call)))

    def to_json(self):
        data = {
            "role": self.role,
            "content": self.content,
        }
        if self.name is not None:
            data["name"] = self.name
        if self.function_call is not None:
            data["function_call"] = self.function_call
        return data

    def message(self) -> "Message":
        return self


class MessageStream:
    def __init__(self, response, final_message=None):
        self.__response = response
        self.__message = {}
        self.__final_message = final_message

    def __next__(self):
        if self.__final_message is not None:
            raise StopIteration()
        chunk = next(self.__response)
        delta = chunk["choices"][0]["delta"]

        def concat(target, delta):
            for k, v in delta.items():
                if isinstance(v, str) or v is None:
                    if k not in target:
                        target[k] = ""
                    target[k] += v or ""
                else:
                    if k not in target:
                        target[k] = {}
                    concat(target[k], v)

        concat(self.__message, delta)
        return delta

    def __iter__(self):
        return self

    def message(self) -> Message:
        if self.__final_message is not None:
            return self.__final_message
        for _ in self:
            ...
        return Message(**self.__message)
