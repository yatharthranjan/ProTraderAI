import copy

from farm import Farm
from tree import Tree
from common.data_retriever import IEXDataRetriever

def build_tree() -> Tree:
    pass


if __name__ == '__main__':
    """Main Entry point of the program"""

    tickers = ['TSLA']

    tree = build_tree()

    # Right now we use the same tree for all ticker symbols. We can supply a custom ticker-tree pair if needed
    ticker_trees = {}
    for ticker in tickers:
        ticker_trees[ticker] = copy.deepcopy(tree)

    data_retriever = IEXDataRetriever()
    farm = Farm(ticker_trees, data_retriever)
    pass
