from tree import Tree
from common.data_retriever import DataRetriever
from accounting import Accountant, AccountantData, SQLiteDatabase
import uuid
import time


class Farm:

    def __init__(self, ticker_trees: dict, data_retriever: DataRetriever, timeframe: str = '5m', accountant: Accountant = Accountant(database=SQLiteDatabase('trades.db')), **kwargs):
        self.ticker_trees = ticker_trees
        self.data_retriever = data_retriever
        self.timeframe = timeframe

    # Iterates over all ticker values sequentially. Make sure to run this in a non blocking thread.
    def iterate(self):
        iter_id = uuid.uuid4()
        for ticker, tree in self.ticker_trees:
            print(f'Getting data for Ticker {ticker}')
            data = self.data_retriever.get_data(ticker, timeframe=self.timeframe)
            print(f'Traversing the Binary Decision tree for ticker')
            current_node = tree.current_node
            while current_node is not None:
                print(f'Got the node => {current_node}. Adding to the accountant')
                event = None
                if current_node.is_leaf():
                    event = current_node.actions

                self.accountant.log(AccountantData(
                    iteration_id=iter_id,
                    ticker=ticker,
                    time=time.time(),
                    decision_node=current_node,
                    event=event
                ))

                # add node to accountant in case we buy or sell
                next_node = tree.traverse(self, current_data=data)
                # run tree.traverse() until receive None
