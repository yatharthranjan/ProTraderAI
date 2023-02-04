# Development moved to fork https://github.com/Ostudiolabs/ProTraderAI
-

## Trading Bot

A simple trading bot based on Binary Decision Trees making it highly customisable - 

### Concepts

1) ***Condition*** - A condition can be any expression that return a boolean. The condition is evaluated for taking binary decision at each branching point (Node).
2) ***Action*** - An action can be any set of instruction that will be executed. For example, sending a notification.
3) ***Node*** - This is the atomic unit in the bot. A node is a branch point of the tree. It consists of 2 nodes (left and right),
a condition and a list of actions.
4) ***Tree*** - This consists of a current node and traverses the tree based on that node. On each traversal, it check the condition
of the current node, if it is true then moves on to left node other moves to the right node. If this is a leaf node, then executes all the actions in the current node.
5) ***Farm*** - A farm consists of a mapping of ticker names and their corresponding Trees. The farm can also iterate over each tree based on the current data.
6) ***Accountant*** - Keeps a log of all the decisions made and any event triggered (like buy or sell).

#### More Info and Usage coming soon
