import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Explore the General Characteristics of the Dataset
def performEDA(df):
    for stockName, data in df.items():
        print(f"--- {stockName} ---")
        print("General info about dataset:")
        print(data.describe())  # Summary statistics for each stock's data
        print("\nMissing Values:")
        print(data.isnull().sum())  # Checking for missing values in the dataset

# Time Series Analysis (Plot Closing Prices for Training and Testing Split)
def plotTimeSeries(df):
    for stockName, data in df.items():
        # Extracting Closing Prices
        closePrices = data['Close']
        
        # Split the data into training and test sets (70% for training, 30% for testing)
        splitIndex = int(len(closePrices) * 0.70)
        trainData = closePrices[:splitIndex]
        testData = closePrices[splitIndex:]
        
        # Plot the training and test data
        plt.figure(figsize=(14, 6))
        plt.plot(trainData, label='Training Set')
        plt.plot(testData, label='Test Set')
        plt.title(f"{stockName} Closing Stock Price")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.show()

# Plot Daily Percentage Change and Volatility
def plotDailyPriceChange(df):
    for stockName, data in df.items():
        # Calculate the daily percentage change
        data['Daily Change (%)'] = data['Close'].pct_change() * 100
        plt.figure(figsize=(14, 6))
        data['Daily Change (%)'].plot(kind='hist', bins=50, title=f"{stockName} Daily Price Change")
        plt.xlabel("Percentage Change")
        plt.ylabel("Frequency")
        plt.show()

# Plot Moving Averages for Stock Prices
def plotMovingAverages(df):
    for stockName, data in df.items():
        # Calculate Short-term (10-day) and Long-term (50-day) Moving Averages
        data['ShortMA'] = data['Close'].rolling(window=10).mean()
        data['LongMA'] = data['Close'].rolling(window=50).mean()
        plt.figure(figsize=(14, 6))
        plt.plot(data['Close'], label='Closing Price')
        plt.plot(data['ShortMA'], label='10 Day Moving Average')
        plt.plot(data['LongMA'], label='50 Day Moving Average')
        plt.title(f"{stockName} Moving Averages")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.show()

# Load CSV Files as DataFrames Using pandas
def getDataFrames(stockList):
    df = {}
    # Load data from CSV files and store them in a dictionary with stock names as keys
    for csv_file in stockList:
        df[csv_file] = pd.read_csv(f"./Veri Toplama/data/{csv_file}.csv", index_col="Date", parse_dates=True)
    return df

def main():
    # Get the List of Stock Names from CSV Files in the Directory
    stockList = [file.replace(".csv", "") for file in os.listdir("./Veri Toplama/data/")]

    # Retrieve DataFrames for Each Stock
    df = getDataFrames(stockList)

    # Perform Exploratory Data Analysis
    performEDA(df)
    
    # Plot Time Series for Stock Prices
    plotTimeSeries(df)

    # Plot Daily Percentage Change in Prices
    plotDailyPriceChange(df)

    # Plot Moving Averages for Stock Prices
    plotMovingAverages(df)

if __name__ == "__main__":
    main()