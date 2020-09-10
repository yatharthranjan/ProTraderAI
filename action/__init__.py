from abc import ABC, abstractmethod


class Action(ABC):

    @abstractmethod
    def execute(self, data, node) -> bool:
        pass
