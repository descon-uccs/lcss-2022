from __future__ import annotations
from abc import abstractmethod, ABC


class Env(ABC):
    @abstractmethod
    def agents(self):
        pass


class Agent(ABC):
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def __str__(self):
        return self.name


class ActionSpace:
    def get_action_set(self, agent: Agent) -> ActionSet:
        pass


class ActionSet:
    pass
