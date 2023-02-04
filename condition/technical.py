import numpy as np
import pandas as pd
import talib as ta
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from condition import Condition
from condition.util import TimeCache

class BaseTechnicalCondition(Condition):
    pass

class TechnicalCondition(BaseTechnicalCondition):

    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.model_cache = TimeCache(refresh_time=86400)

    def check(self, data) -> bool:

        ticker = self.params['ticker']

        stock_data = yf.download(ticker, period='1d') 
        prediction_threshold = data['prediction_threshold']


        # Use the trained model to make a prediction on the current stock price
        X_last = stock_data.iloc[-1, :].drop(columns=['Close']).values

        scaler = StandardScaler()
        # Scale the input data
        X_last = scaler.transform(X_last.reshape(1, -1))

        if self.model_cache.get(ticker) is None:
            model = self.train_model(ticker, data)
            self.model_cache.set(ticker, model)

        model = self.model_cache.get(ticker)

        # Get the prediction
        prediction = model.predict(X_last)

        # Make a decision on whether to buy or sell the stock
        return prediction > prediction_threshold

    def train_model(self, ticker, data):
        stock_data = yf.download(ticker, period=period)

        indicators = data['indicators']

        # Loop through the selected indicators and calculate them using TA-Lib
        for indicator in indicators:
            if indicator == 'SMA':
                stock_data['SMA'] = ta.SMA(stock_data['Close'], timeperiod=14)
            elif indicator == 'EMA':
                stock_data['EMA'] = ta.EMA(stock_data['Close'], timeperiod=14)
            elif indicator == 'RSI':
                stock_data['RSI'] = ta.RSI(stock_data['Close'], timeperiod=14)
            elif indicator == 'ADX':
                stock_data['ADX'] = ta.ADX(stock_data['High'], stock_data['Low'], stock_data['Close'], timeperiod=14)
            elif indicator == 'BBANDS':
                stock_data['BB_UPPER'], stock_data['BB_MIDDLE'], stock_data['BB_LOWER'] = ta.BBANDS(stock_data['Close'], timeperiod=14)

        # Drop any remaining NaN values
        stock_data.dropna(inplace=True)

        # Split the data into training and testing sets
        train_data = stock_data[:int(len(stock_data) * 0.8)]
        test_data = stock_data[int(len(stock_data) * 0.8):]

        # Prepare the training data
        X_train = train_data.drop(columns=['Close']).values
        y_train = (train_data['Close'].shift(-1) > train_data['Close']).astype(int).values

        # Scale the training data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        # Build and compile the model
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Train the model
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

        # Prepare the testing data
        X_test = test_data.drop(columns=['Close']).values
        y_test = (test_data['Close'].shift(-1) > test_data['Close']).astype(int).values

        # Scale the testing data
        X_test = scaler.transform(X_test)

        # Evaluate the model
        score, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print('Test score:', score)
        print('Test accuracy:', accuracy)

        return model


class TechnicalConditionLongTerm(BaseTechnicalCondition):

    def check(self, data) -> bool:

        # Define the stock ticker symbol
        ticker = data['ticker']

        technical_indicators = data['technical_indicators']

        # Get the stock data using yfinance library
        stock_data = yf.Ticker(ticker).history(period="max")

        # Calculate the technical indicators
        for indicator in technical_indicators:
            if indicator == '50d_SMA':
                stock_data["50d_SMA"] = stock_data["Close"].rolling(window=50).mean()
            elif indicator == '200d_SMA':
                stock_data["200d_SMA"] = stock_data["Close"].rolling(window=200).mean()
            elif indicator == 'RSI':
                stock_data["RSI"] = talib.RSI(stock_data["Close"], timeperiod=14)
            elif indicator == 'CCI':
                stock_data["CCI"] = talib.CCI(stock_data["High"], stock_data["Low"], stock_data["Close"], timeperiod=14)
            elif indicator == 'ATR':
                stock_data["ATR"] = talib.ATR(stock_data["High"], stock_data["Low"], stock_data["Close"], timeperiod=14)
        

        # Clean the data by removing any rows with missing values
        stock_data.dropna(inplace=True)

        # Define the target variable
        stock_data["Target"] = np.where(stock_data["Close"].shift(-1) > stock_data["Close"], 1, 0)

        # Split the data into training and test sets
        train_data = stock_data[:int(0.8*len(stock_data))]
        test_data = stock_data[int(0.8*len(stock_data)):]

        # Train a random forest classifier on the training data
        X_train = train_data.drop(["Target", "Close"], axis=1)
        y_train = train_data["Target"]
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)

        # Make predictions on the test data
        X_test = test_data.drop(["Target", "Close"], axis=1)
        y_test = test_data["Target"]
        predictions = clf.predict(X_test)

        # Evaluate the performance of the model
        accuracy = sum(predictions == y_test)/len(y_test)
        print("Accuracy:", accuracy)

        # Get the trading signals for the entire data
        signals = self.generate_signal(stock_data, clf)

        # Add the signals to the dataframe
        stock_data["Signal"] = signals

        # Get the decision for the latest date
        latest_index = len(stock_data) - 1
        decision = self.buy_or_sell(stock_data, latest_index)
        return decision
    
    # Define a function to generate trading signals based on AI
    def generate_signal(self, data, clf):
        X = data.drop(["Target", "Close"], axis=1)
        predictions = clf.predict(X)
        return predictions

    # Define a function to decide whether to buy or sell a stock based on AI
    def buy_or_sell(self, data, index):
        if data["Signal"][index] == 1:
            return True
        else:
            return False
