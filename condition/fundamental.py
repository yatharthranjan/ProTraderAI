import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from condition import Condition

class BaseFundamentalCondition(Condition):
    pass

class FundamentalCondition(BaseFundamentalCondition):

    def check(self, data) -> bool:

        # Define the stock ticker symbol
        ticker = self.params['ticker']

        clf = self.train_model(ticker)

        ## TODO: Change to use correct current data

        # get the current data for the stock from yfinance
        current_earningsPerShare = self.params['earningsPerShare']
        current_priceToBook = self.params['priceToBook']
        current_priceToSales = self.params['priceToSales']
        current_debtToEquity = self.params['debtToEquity']
        current_priceToEarnings = self.params['priceToEarnings']

        prediction = self.predict_buy_sell(clf, current_earningsPerShare, current_priceToBook, current_priceToSales,
                                        current_debtToEquity, current_priceToEarnings)
        
        return prediction == "Buy"

    
    def train_model(self, ticker):
        # Get the stock data using yfinance library
        data = yf.Ticker(ticker).info

        # TODO: Change to use correct historical data

        # Select the features to be used for analysis
        features = ['earningsPerShare', 'priceToBook', 'priceToSales', 'debtToEquity', 'priceToEarnings']

        # Get the target variable - 0 for sell, 1 for buy
        data['target'] = np.where(data['earningsPerShare'] > 0, 1, 0)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(data[features], data['target'], test_size=0.2, random_state=0)

        # Train a Random Forest Classifier
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)

        # Predict on the test data
        y_pred = clf.predict(X_test)

        # Calculate the accuracy of the model
        accuracy = clf.score(X_test, y_test)
        print("Accuracy: ", accuracy)

        return clf


    def predict_buy_sell(self, clf, earningsPerShare, priceToBook, priceToSales, debtToEquity, priceToEarnings):
        # Predict the target for the given features
        prediction = clf.predict([[earningsPerShare, priceToBook, priceToSales, debtToEquity, priceToEarnings]])
        if prediction == 1:
            return "Buy"
        else:
            return "Sell"