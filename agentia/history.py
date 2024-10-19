from typing import Any, Optional
from .message import Message


class History:
    def __init__(self, instructions: Optional[str]) -> None:
        self.__instructions = instructions
        self.__messages: list[Message] = []
        self.reset()

    def get(self) -> list[Message]:
        return list(self.__messages)

    def reset(self):
        self.__messages = []
        if self.__instructions is not None:
            self.add(Message(role="system", content=self.__instructions))

    def add(self, message: Message):
        # TODO: auto trim history
        self.__messages.append(message)

    def get_raw_messages(self) -> Any:
        return [m.to_dict() for m in self.__messages]

    def set_raw_messages(self, data: Any):
        self.__messages = [Message.from_dict(m) for m in data]
