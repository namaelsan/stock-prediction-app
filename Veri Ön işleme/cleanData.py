# eksik ve anormal değerleri sil
# hareketleri ortalamaları bul (10 ve 50 günlük)
# min-max scaling yap
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os

def minmaxScale(prices):
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    prices_scaled = scaler.fit_transform(prices)
    return prices_scaled



if __name__ == "__main__":
    csv_files = []

    for file in os.listdir("./Veri Toplama/data/"):
        csv_files.append(file)
    
    for csv_file in csv_files:
        dataFrame = pd.read_csv(f"./Veri Toplama/data/{csv_file}")
        prices = dataFrame['Close'].values.reshape(-1, 1)
        prices = prices[2:]
        prices = minmaxScale(prices)
        
    
    exit()