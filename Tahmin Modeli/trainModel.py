import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import SGD
import keras.models
import math
from sklearn.metrics import mean_squared_error


def initModel():
    regressor = Sequential()
    # First LSTM layer with Dropout regularisation
    regressor.add(LSTM(units=50, return_sequences=True, input_shape=(pastDays,feature)))
    regressor.add(Dropout(0.3))
    # Second LSTM layer
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.3))
    # Third LSTM layer
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.5))
    # Fourth LSTM layer
    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.5))
    # The output layer
    regressor.add(Dense(units=1))

    # Compiling the RNN
    regressor.compile(optimizer='rmsprop', loss='mean_squared_error')
    return regressor

def splitTrainAndTest(df,csv_files,trainPercentage):
    df_new={}
    for i in csv_files:
        df_new[i] = {}
        time = (int) ((len(df[i]) -1) * trainPercentage)

        if (len(df[i])-1 - time) < pastDays:
            splitTime= df[i].index[len(df[i])-1 -pastDays]
        else:
            splitTime= df[i].index[time]
        df_new[i]["Train"]= df[i].loc[:splitTime,["Close","Volume"]]
        df_new[i]["Test"] = df[i].loc[splitTime:,["Close","Volume"]]
    return df_new


def splitDataXy(transform_train,csv_files,trainScalers):
    trainset = {}
    for j in csv_files:
        trainset[j] = {}
        X_train = []
        y_train = []
        for i in range(pastDays,len(transform_train[j]) - futureDays):
            X_train.append(transform_train[j][i - pastDays:i,[0,1]])
            y_train.append(transform_train[j][i + futureDays,0])

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        y_train = trainScalers[j].fit_transform(y_train.reshape(-1,1))
        trainset[j]["X"] = X_train
        trainset[j]["y"] = y_train 
    return trainset
    
def regressorFit(csv_files,regressor,trainset):
    for i in csv_files:
        print("Fitting to", i)
        history = regressor.fit(trainset[i]["X"], trainset[i]["y"], epochs=10, batch_size=150)
    return history 


def scaleData(csv_files,df_new):
    transform_train = {}
    trainScalers = {}
    for i in csv_files:
        trainSc = MinMaxScaler(feature_range=(0,1))

        # Reshape the Close column and apply MinMaxScaler
        a0=df_new[i]["Train"].to_numpy()

        transform_train[i] = trainSc.fit_transform(a0)
        trainScalers[i] = trainSc

    return transform_train, trainScalers

def csvNameList():
    csv_files = []
    for file in os.listdir("./Veri Toplama/data/"):
        csv_files.append(file.replace(".csv",""))
    return csv_files

def getDataFrames(csv_files):
    df = {}
    for csv_file in csv_files:
        df[csv_file] = pd.read_csv(f"./Veri Toplama/data/{csv_file}.csv",index_col="Date",parse_dates=True)
    return df

def main():

    csv_files = csvNameList()

    df = getDataFrames(csv_files)

    df_new = splitTrainAndTest(df,csv_files,trainPercentage=0.7)

    transform_train, trainScalers = scaleData(csv_files,df_new)

    del df_new

    trainset = splitDataXy(transform_train,csv_files,trainScalers)

    regressor = initModel()

    lossHistory = regressorFit(csv_files,regressor,trainset)

    regressor.save("Ağırlıklar/test.keras")


if __name__ == "__main__":
    futureDays=1
    pastDays=60
    feature=2
    main()