
# Import necessary libraries
import pandas as pd
from condition import Condition

class PortfolioCondition(Condition):
    pass

class AverageDownCondition(PortfolioCondition):
    def __init__(self, condition):
        self.condition = condition

    def check(self, data) -> bool:
        ticker = self.params['ticker']
        positions = self.params['positions']
        threshold = self.params['threshold']

        # Define the threshold for loss
        threshold = 0.05

        # Load the data into a DataFrame
        df = pd.DataFrame(positions)

        # Iterate through all the tickers in the DataFrame
        for ticker in df["ticker"].unique():
            # Filter the DataFrame to only include the current ticker
            ticker_df = df[df["ticker"] == ticker]
            
            # Calculate the average cost per share for the ticker
            avg_cost = ticker_df["cost"].sum() / ticker_df["shares"].sum()
            
            # Calculate the current price per share for the ticker
            curr_price = ticker_df["price"].iloc[0]
            
            # Calculate the percentage loss
            percent_loss = (avg_cost - curr_price) / avg_cost
            
            # If the percentage loss is over the threshold, buy more shares to average down
            return percent_loss > threshold




                ## TODO: Move this to buy action

                # # Calculate the number of shares to buy
                # shares_to_buy = (threshold / percent_loss) * ticker_df["shares"].sum()
                
                # # Add the new shares to the existing shares
                # ticker_df["shares"] = ticker_df["shares"] + shares_to_buy
                
                # # Update the cost per share to reflect the average down
                # ticker_df["cost"] = (ticker_df["cost"].sum() + (shares_to_buy * curr_price)) / ticker_df["shares"].sum()
                
                # # Replace the DataFrame for the current ticker with the updated DataFrame
                # df.loc[df["ticker"] == ticker] = ticker_df

                # Save the updated DataFrame to a CSV file
        # df.to_csv("open_positions_updated.csv", index=False)