import tensorflow as tf
import numpy as np
import pandas as pd
import requests
import json
from condition import Condition
from condition.util import TimeCache

class SentimentCondition(Condition):
    pass

class MarketSentimentCondition(SentimentCondition):

    def __init__(self, params, news_api_key, period=None):
        super().__init__(params)
        self.news_api_key = news_api_key

        if period is None:
            self.period = 5

        self.start_date = pd.Timestamp.now(tz='utc') -  pd.Timedelta(years=period)
        self.end_date = pd.Timestamp.now(tz='utc')
        self.model_cache = TimeCache(max_size=1000, refresh_time=86400)
    

    def get_news_sentiment(self, ticker):
        # Get the latest news articles related to the ticker
        news_api_key = self.news_api_key
        url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={news_api_key}&from={self.start_date.date}&to={self.end_date.date}"
        response = requests.get(url)
        data = json.loads(response.text)
        
        # Calculate the sentiment of each article using a sentiment analysis library
        sentiment = []
        for article in data['articles']:
            sentiment.append(self.calculate_sentiment(article['description']))
            
        return sentiment

    def calculate_sentiment(self, text):
        # Use a sentiment analysis library to calculate the sentiment of the text
        # Example using the Vader Sentiment library:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        sentiment = analyzer.polarity_scores(text)['compound']
        
        return sentiment

    def prepare_data(self, ticker, window):
        # Get the market sentiment data for the given ticker
        sentiment = self.get_news_sentiment(ticker)
        
        # Create a DataFrame to store the sentiment values
        data = pd.DataFrame({'sentiment': sentiment})
        
        # Create the features by shifting the sentiment values
        for i in range(1, window + 1):
            data[f'sentiment_{i}'] = data['sentiment'].shift(i)
            
        # Drop any rows with missing data
        data.dropna(inplace=True)
        
        # Create the target variable by indicating if the stock price increased or decreased
        data['target'] = np.where(data['sentiment'] > 0, 1, 0)
        
        return data

    def create_transformer_model(self, window, feature_count, heads, size, activation="relu"):
        input_layer = tf.keras.layers.Input(shape=(window, feature_count))
        transformer = tf.keras.layers.Transformer(
            num_heads=heads,
            size_per_head=size,
            activation=activation
        )(input_layer)
        output_layer = tf.keras.layers.Dense(1, activation="sigmoid")(transformer)
        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        return model

    def train_model(self, model, X_train, y_train, epochs):
        # Compile the model
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        
        # Train the model
        model.fit(X_train, y_train, epochs=epochs)
        
        return model


    def check(self, data) -> bool:
        ticker=self.params['ticker']
        window=self.params['window']

        if self.model_cache.get(ticker) is None:
            # Get the model
            model = self.create_transformer_model(window, 1, 4, 32)

            # Prepare the data
            data = self.prepare_data(ticker, window)
            X_train = data.drop('target', axis=1).values
            y_train = data['target'].values

            # Train the model
            model = self.train_model(model, X_train, y_train, 10)

            # Save the model
            self.model_cache.set(ticker, model)

        model = self.model_cache.get(ticker)

        # Get the latest market sentiment data
        sentiment = self.get_news_sentiment(ticker)
        
        # Prepare the data in the same way as during training
        latest_data = pd.DataFrame({'sentiment': sentiment})
        for i in range(1, window + 1):
            latest_data[f'sentiment_{i}'] = latest_data['sentiment'].shift(i)
        latest_data.dropna(inplace=True)
        X_test = latest_data.values
        X_test = np.expand_dims(X_test, axis=0)
        
        # Use the trained model to make a prediction
        y_prob = model.predict(X_test)
        y_pred = np.round(y_prob)
        
        # Decide whether to buy or sell the stock based on the prediction
        return y_pred == 1