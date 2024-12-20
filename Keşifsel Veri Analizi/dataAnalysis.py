import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# 1. Veri Setinin Genel Özelliklerini Anlama
def performEDA(df):
    for stockName, data in df.items():
        print(f"--- {stockName} ---")
        print("General info about dataset:")
        print(data.describe())  # Özet istatistikler
        print("\nMissing Values:")
        print(data.isnull().sum())  # Eksik değer kontrolü

# 2. Zaman Serisi Analizi
def plotTimeSeries(df):
    for stockName, data in df.items():
        # Kapanış Priceları
        closePrices = data['Close']
        
        # Veriyi yüzde 70 ve yüzde 30 olarak bölmek
        splitIndex = int(len(closePrices) * 0.70)
        trainData = closePrices[:splitIndex]
        testData = closePrices[splitIndex:]
        
        # Grafiği çizmek
        plt.figure(figsize=(14, 6))
        plt.plot(trainData, label='Training Set')
        plt.plot(testData, label='Test Set')
        plt.title(f"{stockName} Closing Stock Price")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.show()



# 3. Günlük Yüzdesel Değişim ve Volatilite
def plotDailyPriceChange(df):
    for stockName, data in df.items():
        data['Daily Change (%)'] = data['Close'].pct_change() * 100
        plt.figure(figsize=(14, 6))
        data['Daily Change (%)'].plot(kind='hist', bins=50, title=f"{stockName} Daily Price Change")
        plt.xlabel("Percentage Change")
        plt.ylabel("Frequency")
        plt.show()

# 4. Hareketli Ortalamalar
def plotMovingAverages(df):
    for stockName, data in df.items():
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

# 5. Korelasyon Analizi
def plotCorrelationMatrix(df):
    for stockName, data in df.items():
        correlation_matrix = data[['Close', 'Volume']].corr()
        plt.figure(figsize=(6, 4))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title(f"{stockName} Correlation Matrix")
        plt.show()


# Retrieves the csv files as dataFrames with pandas library
def getDataFrames(stockList):
    df = {}
    for csv_file in stockList:
        df[csv_file] = pd.read_csv(f"./Veri Toplama/data/{csv_file}.csv", index_col="Date", parse_dates=True)
    return df

def main():
    # Get a list of the stocks present
    stockList = [file.replace(".csv", "") for file in os.listdir("./Veri Toplama/data/")]

    # Retrieve the dataFrames from the list
    df = getDataFrames(stockList)

    performEDA(df)
    plotTimeSeries(df)
    plotDailyPriceChange(df)
    plotMovingAverages(df)
    plotCorrelationMatrix(df)

if __name__ == "__main__":
    main()