from typing import Optional
from .message import Message, Role


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
            self.add(Message(role=Role.SYSTEM, content=self.__instructions))

    def add(self, message: Message):
        # TODO: auto trim history
        self.__messages.append(message)
