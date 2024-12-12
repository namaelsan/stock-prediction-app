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


def getModel():
    if(os.path.exists("Ağırlıklar/test.keras")):
        print("model already exists")
        regressor = keras.models.load_model('Ağırlıklar/test.keras')
        return regressor
    else:
        return None

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


def splitDataXy(transform_train,csv_files,testScalers):
    testset = {}
    for j in csv_files:
        testset[j] = {}
        X_test = []
        y_test = []
        for i in range(pastDays,len(transform_train[j]) - futureDays +1):
            X_test.append(transform_train[j][i - pastDays:i,[0,1]])
            y_test.append(transform_train[j][i + futureDays -1,0])

        X_test = np.array(X_test)
        y_test = np.array(y_test)
        y_test = testScalers[j]["y"].fit_transform(y_test.reshape(-1,1))

        X_shape = X_test.shape
        X_test = testScalers[j]["X"].fit_transform(X_test.reshape(-1,feature))
        X_test = np.array(X_test).reshape(X_shape)
        testset[j]["X"] = X_test
        testset[j]["y"] = y_test 
    return testset
    

def scaleData(csv_files,df_new):
    ####scaling gets done later on
    transform_test = {}
    testScalers = {}
    for i in csv_files:
        testScalers[i]= {}

        a1=df_new[i]["Test"].to_numpy()
        transform_test[i] = a1

        testScalers[i]["X"] = MinMaxScaler(feature_range=(0,1))
        testScalers[i]["y"] = MinMaxScaler(feature_range=(0,1))
    return transform_test, testScalers

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

    df_new = splitTrainAndTest(df,csv_files,0.7)

    transform_test, testScalers = scaleData(csv_files,df_new)

    testset = splitDataXy(transform_test,csv_files,testScalers)

    regressor = getModel()
    
    if(regressor == None):
        print("no model found")
        exit()

    totalPercentage=0
    pred_result = {}
    for i in csv_files:
        y_true = testScalers[i]["y"].inverse_transform(testset[i]["y"].reshape(-1,1))
        prediction=regressor.predict(testset[i]["X"])
        y_pred = testScalers[i]["y"].inverse_transform(prediction)
        MSE = mean_squared_error(y_true, y_pred)
        pred_result[i] = {}
        pred_result[i]["True"] = y_true
        pred_result[i]["Pred"] = y_pred
        
        success=0
        fail=0

        for j in range(0,len(y_pred)):
            pastDaysLayer=(int) (j/pastDays)
            remainingDaysLayer=(int) (j%pastDays)
            pastX=np.stack(testset[i]["X"])

            shape1 =(pastX.shape[0],pastX.shape[1],feature)
            pastXShaped = testScalers[i]["X"].inverse_transform(pastX.reshape(-1,feature))
            pastTransformed = np.reshape(pastXShaped, shape1)


            past=pastTransformed[j][59][0]
            pred=y_pred[j][0]
            true=y_true[j][0]
            if ((pred - past) * (true - past)) > 0:
                print("GREAT SUCCESS")
                success+=1
            else:
                print("lowlife fail")
                fail+=1
            print(f"Prediction= {pred}---True= {true}---Past= {past}")

            a=y_pred[j][0]
            b=testset[i]["X"][j][0][0]


        print(f"Succes Rate:{(success/(success+fail)):.2f}")
        totalPercentage+=success/(success+fail)

        plt.figure(figsize=(14,6))
        plt.title(f"{i} with MSE {MSE}")
        plt.plot(y_true)
        plt.plot(y_pred,".")
        plt.show()
    print(f"Overall Success:{totalPercentage/len(csv_files)}")


if __name__ == "__main__":
    futureDays=1
    pastDays=60
    feature=2
    main()