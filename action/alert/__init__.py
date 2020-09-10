from action import Action


class AlertAction(Action):

    def execute(self, data, node) -> bool:
        try:
            self.notify()
            return True
        except:
            return False

    def notify(self):
        pass
