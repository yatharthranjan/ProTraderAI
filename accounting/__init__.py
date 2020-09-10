from enum import Enum


class Event(Enum):
    BUY = "BUY",
    SELL = "SELL"


class AccountantData:
    def __init__(self, iteration_id, ticker, time, decision_node, event: [Event, None] = None):
        self.iteration_id = iteration_id
        self.ticker = ticker
        self.time = time
        self.decision_node = decision_node
        self.event = event


class Accountant:

    def __init__(self):
        self.data = AccountantData
        pass

    def log(self, accountant_data: AccountantData):
        pass

    def load(self) -> list:
        pass
