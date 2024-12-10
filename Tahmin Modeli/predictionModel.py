import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from keras.optimizers import SGD
import math
from sklearn.metrics import mean_squared_error


def split(dataframe,border,col):
    return dataframe.loc[:border,col], dataframe.loc[border:,col]


csv_files = []

for file in os.listdir("./Veri Toplama/data/"):
    csv_files.append(file.replace(".csv",""))

df = {}
for csv_file in csv_files:
  df[csv_file] = pd.read_csv(f"./Veri Toplama/data/{csv_file}.csv",index_col="Date",parse_dates=["Date"])

df_new = {}

for i in csv_files:
    df_new[i] = {}
    time = (int) ((len(df[i]) -1) * 0.3)
    splitTime= df[i].index[time]

    df_new[i]["Train"]= df[i].loc[:splitTime,"Close"]
    df_new[i]["Test"] = df[i].loc[splitTime:,"Close"]
    print(df_new[i]["Train"])
    print("\n")
    print(df_new[i]["Test"])

transform_train = {}
transform_test = {}
scaler = {}

for num, i in enumerate(csv_files):
  sc = MinMaxScaler(feature_range=(0,1))
  a0 = np.array(df_new[i]["Train"])
  a1 = np.array(df_new[i]["Test"])
  a0 = a0.reshape(a0.shape[0],1)
  a1 = a1.reshape(a1.shape[0],1)

  transform_train[i]=sc.fit_transform(a0)
  transform_test[i]=sc.fit_transform(a1)
  scaler[i]=sc

del a0
del a1


# for i in transform_train.keys():
#     print(i,transform_train[i].shape)
# print("\n")
# for i in transform_test.keys():
#     print(i,transform_test[i].shape)

trainset = {}
testset = {}
for j in csv_files:
    trainset[j] = {}
    X_train = []
    y_train = []
    for i in range(7,len(transform_train[j])):
        print()
        X_train.append(transform_train[j][i-7:i,0])
        y_train.append(transform_train[j][i,0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    print(X_train)
    val =(X_train.shape[0],X_train.shape[1],1)
    print(val)
    trainset[j]["X"] = np.reshape(X_train, val)
    trainset[j]["y"] = y_train
    
    testset[j] = {}
    X_test = []
    y_test = []    
    for i in range(7, len(transform_test[j])):
        X_test.append(transform_test[j][i-7:i,0])
        y_test.append(transform_test[j][i,0])
    X_test, y_test = np.array(X_test), np.array(y_test)
    testset[j]["X"] = np.reshape(X_test, (X_test.shape[0], X_train.shape[1], 1))
    testset[j]["y"] = y_test

arr_buff = []
for i in csv_files:
    buff = {}
    buff["X_train"] = trainset[i]["X"].shape
    buff["y_train"] = trainset[i]["y"].shape
    buff["X_test"] = testset[i]["X"].shape
    buff["y_test"] = testset[i]["y"].shape
    arr_buff.append(buff)


# The LSTM architecture
regressor = Sequential()
# First LSTM layer with Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
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
    regressor.fit(trainset[i]["X"], trainset[i]["y"], epochs=10, batch_size=200)

pred_result = {}
for i in csv_files:
    y_true = scaler[i].inverse_transform(testset[i]["y"].reshape(-1,1))
    y_pred = scaler[i].inverse_transform(regressor.predict(testset[i]["X"]))
    MSE = mean_squared_error(y_true, y_pred)
    pred_result[i] = {}
    pred_result[i]["True"] = y_true
    pred_result[i]["Pred"] = y_pred
    
    plt.figure(figsize=(14,6))
    plt.title("{} with MSE {:10.4f}".format(i,MSE))
    plt.plot(y_true)
    plt.plot(y_pred)
    plt.show()