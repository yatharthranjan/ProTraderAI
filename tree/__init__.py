from node import Node


class Tree:
    def __init__(self, root_node: Node):
        self.root_node = root_node
        self.current_node = root_node

    def traverse(self, current_data) -> [Node, None]:
        if self.current_node.condition.check(current_data):
            if self.current_node.left is not None:
                self.current_node = self.current_node.left
                return self.current_node
        else:
            if self.current_node.right is not None:
                self.current_node = self.current_node.right
                return self.current_node

            # Only execute actions if the node is a leaf
            if self.current_node.left is None and self.current_node.right is None:
                if self.current_node.actions is not None:
                    for action in self.current_node.actions:
                        action.execute(current_data, self)
                self.current_node = self.root_node

        return None
