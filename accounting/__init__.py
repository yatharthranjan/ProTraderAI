class AccountantData:
    def __init__(self, iteration_id, ticker, time, decision_node, event: [str, None] = None):
        self.iteration_id = iteration_id
        self.ticker = ticker
        self.time = time
        self.decision_node = decision_node
        self.event = event


class Accountant:

    def __init__(self):
        self.data = self.load()
        pass

    def log(self, accountant_data: AccountantData):
        pass

    def load(self) -> list:
        pass
