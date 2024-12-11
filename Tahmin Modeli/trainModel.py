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




def split(dataframe,border,col):
    return dataframe.loc[:border,col], dataframe.loc[border:,col]

if __name__ == "__main__":

    futureDays=1
    pastDays=60
    feature=2

    csv_files = []
    for file in os.listdir("./Veri Toplama/data/"):
        csv_files.append(file.replace(".csv",""))

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
        df_new[i]["Train"]= df[i].loc[:splitTime,["Close","Volume"]]
        df_new[i]["Test"] = df[i].loc[splitTime:,["Close","Volume"]]  


    transform_train = {}
    transform_test = {}
    trainScalers = {}
    testScalers = {}


    for num, i in enumerate(csv_files):
        trainSc = MinMaxScaler(feature_range=(0,1))
        testScalers[i]= {}

        # Reshape the Close column and apply MinMaxScaler
        a0=df_new[i]["Train"].to_numpy()
        a1=df_new[i]["Test"].to_numpy()

        transform_train[i] = trainSc.fit_transform(a0)
        transform_test[i] = a1
        trainScalers[i] = trainSc
        testScalers[i]["X"] = MinMaxScaler(feature_range=(0,1))
        testScalers[i]["y"] = MinMaxScaler(feature_range=(0,1))

    del a0
    del a1


    trainset = {}
    testset = {}

    for j in csv_files:
        trainset[j] = {}
        X_train = []
        y_train = []
        for i in range(pastDays,len(transform_train[j]) - futureDays +1):
            X_train.append(transform_train[j][i - pastDays:i,[0,1]])
            y_train.append(transform_train[j][i + futureDays -1,0])

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        shape1 =(X_train.shape[0],X_train.shape[1],feature)

        y_train = trainScalers[j].fit_transform(y_train.reshape(-1,1))
        trainset[j]["X"] = np.reshape(X_train, shape1)
        trainset[j]["y"] = y_train 


        testset[j] = {}
        X_test = []
        y_test = []    
        for i in range(pastDays, len(transform_test[j]) - futureDays +1):
            X_test.append(transform_test[j][i - pastDays:i,[0,1]])
            y_test.append(transform_test[j][i + futureDays -1,0])

        X_test, y_test = np.stack(X_test), np.array(y_test)
        X_shape = X_test.shape
        y_test = testScalers[j]["y"].fit_transform(y_test.reshape(-1,1))
        X_test = testScalers[j]["X"].fit_transform(X_test.reshape(-1,feature))
        X_test = np.array(X_test).reshape(X_shape)

        shape2 =(X_test.shape[0],X_test.shape[1],2)
        testset[j]["X"] = X_test
        testset[j]["y"] = y_test

    arr_buff = []
    for i in csv_files:
        buff = {}
        buff["X_train"] = trainset[i]["X"].shape
        buff["y_train"] = trainset[i]["y"].shape
        buff["X_test"] = testset[i]["X"].shape
        buff["y_test"] = testset[i]["y"].shape
        arr_buff.append(buff)


    # regressor= {}

    if(os.path.exists("Ağırlıklar/test.keras")):
        print("model already exists")
        regressor = keras.models.load_model('Ağırlıklar/test.keras')
    else:
        # The LSTM architecture
        regressor = Sequential()
        # First LSTM layer with Dropout regularisation
        regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],feature)))
        print(X_train.shape)
        regressor.add(Dropout(0.2))
        # Second LSTM layer
        regressor.add(LSTM(units=50, return_sequences=True))
        regressor.add(Dropout(0.2))
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
        # Fitting to the training set
        for i in csv_files:
            print("Fitting to", i)
            history = regressor.fit(trainset[i]["X"], trainset[i]["y"], epochs=10, batch_size=200)
            print(trainset[i]["X"])
            # lossPerEpoch = regressor[i].history.history["loss"]
            # plt.plot(range(len(lossPerEpoch)),lossPerEpoch)
            # plt.show()

    regressor.save("Ağırlıklar/test.keras")


    pred_result = {}
    for i in csv_files:
        y_true = testScalers[i]["y"].inverse_transform(testset[i]["y"].reshape(-1,1))
        prediction=regressor.predict(testset[i]["X"])
        y_pred = testScalers[i]["y"].inverse_transform(prediction)
        MSE = mean_squared_error(y_true, y_pred)
        pred_result[i] = {}
        pred_result[i]["True"] = y_true
        pred_result[i]["Pred"] = y_pred
    
        plt.figure(figsize=(14,6))
        plt.title("{} with MSE {:10.4f}".format(i,MSE))
        plt.plot(y_true)
        plt.plot(y_pred)
        plt.show()