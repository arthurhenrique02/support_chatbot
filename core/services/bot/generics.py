import typing
from abc import ABC, abstractmethod


class GenericBot(ABC):
    name: str
    model: str
    llm: typing.Any
    definition_prompt: str

    def __init__(self, name: str, definition_prompt: str, model: str):
        self.name = name
        self.definition_prompt = definition_prompt
        self.llm = None
        self.model = model

    @abstractmethod
    def task(self, prompt: str, *args, **kwargs) -> str:
        raise NotImplementedError

    @abstractmethod
    def setup_llm(self, *args, **kwargs) -> None:
        raise NotImplementedError
