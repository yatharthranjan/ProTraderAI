from abc import ABC, abstractmethod
import pandas as pd


class DataRetriever(ABC):

    @abstractmethod
    def get_data(self, ticker: str, timeframe: str) -> pd.DataFrame:
        pass


class IEXDataRetriever(DataRetriever):

    def __init__(self):
        return

    def get_data(self, ticker: str, timeframe: str) -> pd.DataFrame:
        pass
