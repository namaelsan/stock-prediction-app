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

def formatData(csv_files):
    try:
        for csv_file in csv_files:
            with open(f"./Veri Toplama/data/{csv_file}", "r") as fr:
                # reading line by line
                lines = fr.readlines()                
                lines[0] = lines[0].replace("Price", "Date", 1)
                
                # pointer for position
                ptr = 1


                # opening in writing mode
                with open(f"./Veri Toplama/data/{csv_file}", "w") as fw:
                    for line in lines:
                        if ptr != 2 and ptr != 3:
                            fw.write(line)
                        ptr += 1

        print("Formatted")
    except:
        print("Formatting error")


def removingUselessData(csv_files):
    for csv_file in csv_files:
        dataFrame = pd.read_csv(f"./Veri Toplama/data/{csv_file}")
        prices = dataFrame['Close'].values.reshape(-1, 1)
        columnsToKeep = ['Date','Close','Volume']
        dataFrame = dataFrame[columnsToKeep]
        if(csv_file == "LMYK.csv" or csv_file == "ALTNY.csv" or csv_file == "LMKDC.csv" or csv_file == "OBAMS.csv"):
            os.remove(f"./Veri Toplama/data/{csv_file}")
            continue
        dataFrame.to_csv(f"./Veri Toplama/data/{csv_file}",index=False)


if __name__ == "__main__":
    csv_files = []

    for file in os.listdir("./Veri Toplama/data/"):
        csv_files.append(file)
    formatData(csv_files)
    removingUselessData(csv_files)
    
    exit()