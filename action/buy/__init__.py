from action import Action

class BuyAction(Action):

    def execute(self, data, node) -> bool:
        try:
            self.buy(data)
            return True
        except:
            return False

    def buy(self, data):
        pass