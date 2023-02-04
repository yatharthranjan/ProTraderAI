from abc import ABC, abstractmethod


class Condition(ABC):

    def __init__(self, params, **kwargs):
        self.params = params

    @abstractmethod
    def check(self, data) -> bool:
        pass
