import copy

from farm import Farm
from tree import Tree
from common.data_retriever import IEXDataRetriever
from node import Node
from condition.fundamental import FundamentalCondition, TechnicalCondition, SentimentCondition
from condition.portfolio import AverageDownCondition
from action.buy import BuyAction

def build_tree(params) -> Tree:

    fun_cond = FundamentalCondition(params=params)
    tech_cond = TechnicalCondition(params=params)
    sent_cond = SentimentCondition(params=params, news_api_key="", period=5)
    buy_act =  BuyAction(params)
    sent_node = Node("Sentiment Analysis", sent_cond, None, None, buy_act)
    tech_node = Node("Technical Analysis", tech_cond, sent_node, None, None)
    fun_node = Node("Fundamental Analysis", fun_cond, tech_node, None, None)

    tree = Tree(fun_node)
    
    return tree

def build_tree_average_down(params) -> Tree:
    avg_cond = AverageDownCondition(params)
    buy_act = BuyAction(params)
    avg_down_node = Node("Average Down", avg_cond, None, None, buy_act)

    tree = Tree(avg_down_node)
    
    return tree

if __name__ == '__main__':
    """Main Entry point of the program"""

    params = {
        'ticker': 'TSLA',
    }
    # Right now we use the same tree for all ticker symbols. We can supply a custom ticker-tree pair if needed
    ticker_trees = {
        'TSLA': build_tree(params=params),
        'MSFT': build_tree_average_down(params=params),
    }

    data_retriever = IEXDataRetriever()
    farm = Farm(ticker_trees, data_retriever)
    
    node = ""
    while(node is not None):
        node = farm.iterate()
