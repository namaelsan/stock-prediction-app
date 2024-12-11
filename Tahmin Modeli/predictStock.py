import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from keras.optimizers import SGD
import keras
import math
from sklearn.metrics import mean_squared_error


if __name__ == "__main__":

    futureDays=1
    pastDays=45

    csv_files = []
    keras_files = []

    for file in os.listdir("./Veri Toplama/data/"):
        csv_files.append(file.replace(".csv",""))

    for file in os.listdir("./Ağırlıklar/"):
        keras_files.append(file.replace(".keras",""))

    transform_train = {}
    transform_test = {}
    scaler = {}

    df = {}
    for csv_file in csv_files:
        df[csv_file] = pd.read_csv(f"./Veri Toplama/data/{csv_file}.csv",index_col="Date",parse_dates=True)

    df_new = {}

    for i in csv_files:
        df_new[i] = {}
        time = (int) ((len(df[i]) -1) * 0.7)

        if (len(df[i])-1 - time) < pastDays:
            splitTime= df[i].index[len(df[i])-1 -pastDays]
        else:
            splitTime= df[i].index[time]
        df_new[i]["Train"]= df[i].loc[:splitTime,"Close"]
        df_new[i]["Test"] = df[i].loc[splitTime:,"Close"]

    for num, i in enumerate(csv_files):
        sc0 = MinMaxScaler(feature_range=(0,1))
        sc1 = MinMaxScaler(feature_range=(0,1))
        a0 = np.array(df_new[i]["Train"])
        a1 = np.array(df_new[i]["Test"])
        a0 = a0.reshape(a0.shape[0],1)
        a1 = a1.reshape(a1.shape[0],1)

        transform_train[i]=sc0.fit_transform(a0)
        transform_test[i]=sc1.fit_transform(a1)
        scaler[i]=sc1

    trainset = {}
    testset = {}
    for j in csv_files:
        trainset[j] = {}
        X_train = []
        y_train = []
        for i in range(pastDays,len(transform_train[j]) - futureDays +1):
            X_train.append(transform_train[j][i - pastDays:i,0])
            y_train.append(transform_train[j][i + futureDays -1,0])


        X_train = np.array(X_train)
        y_train =  np.array(y_train)
        val =(X_train.shape[0],X_train.shape[1],1)
        trainset[j]["X"] = np.reshape(X_train, val)
        trainset[j]["y"] = y_train

        testset[j] = {}
        X_test = []
        y_test = []    
        for i in range(pastDays, len(transform_test[j]) - futureDays +1):
            X_test.append(transform_test[j][i - pastDays:i,0])
            y_test.append(transform_test[j][i + futureDays -1,0])

        X_test, y_test = np.array(X_test), np.array(y_test)
        testset[j]["X"] = np.reshape(X_test, (X_test.shape[0], X_train.shape[1], 1))
        testset[j]["y"] = y_test

    if(len(keras_files) == 1):
        regressor = keras.models.load_model('Ağırlıklar/test.keras')

        pred_result = {}
        for i in csv_files:
            if testset[i]["y"].reshape(-1,1).size == 0:
                print(f"Warning: No test data for {i}")
            else:
                y_true = scaler[i].inverse_transform(testset[i]["y"].reshape(-1, 1))

            # print(testset[i]["y"].reshape(-1,1))
            # print(testset[i]["y"])

            y_true = scaler[i].inverse_transform(testset[i]["y"].reshape(-1,1))
            y_pred = scaler[i].inverse_transform(regressor.predict(testset[i]["X"]))
            MSE = mean_squared_error(y_true, y_pred)
            pred_result[i] = {}
            pred_result[i]["True"] = y_true
            pred_result[i]["Pred"] = y_pred
            
            plt.figure(figsize=(14,6))
            plt.title("{} with MSE {:10.4f}".format(i,MSE))
            plt.plot(y_true)
            plt.plot(y_pred,'.')
            plt.show()
    elif(len(keras_files) > 1):
        for keras_file in keras_files:
            regressor = keras.models.load_model(f'Ağırlıklar/{keras_file}')
            y_true = scaler[i].inverse_transform(testset[i]["y"].reshape(-1,1))
            y_pred = scaler[i].inverse_transform(regressor.predict(testset[i]["X"]))
            MSE = mean_squared_error(y_true, y_pred)
            pred_result[i] = {}
            pred_result[i]["True"] = y_true
            pred_result[i]["Pred"] = y_pred
            
            plt.figure(figsize=(14,6))
            plt.title("{} with MSE {:10.4f}".format(i,MSE))
            plt.plot(y_true)
            plt.plot(y_pred,'o')
            plt.show()
    else:
        print("no model found")
