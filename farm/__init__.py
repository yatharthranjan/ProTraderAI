from tree import Tree
import copy


class Farm:

    def __init__(self, tree: Tree, tickers: list):
        self.ticker_trees = {}
        for ticker in tickers:
            self.ticker_trees[ticker] = copy.deepcopy(tree)

    def iterate(self):
        for ticker, tree in self.ticker_trees:
            print(f'Getting data for Ticker {ticker}')
            # get data for ticker
            print(f'Traversing the Binary Decision tree for ticker')
            # run tree.traverse() until receive None
            print(f'Got the node => . Adding to accountant')
            # add node to accountant in case we buy or sell
            pass
