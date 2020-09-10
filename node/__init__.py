from condition import Condition

class Node:
    def __init__(self, name: str, condition: Condition, left=None, right=None, actions: list=None):
        self.left = left
        self.right = right
        self.condition = condition
        self.actions = actions
        self.name = name

