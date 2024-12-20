import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import SGD, Adam
import keras
from keras import regularizers
import math
from PIL import Image

# Set constants for future prediction, past history and features
futureDays = 1
pastDays = 60
feature = 2

# Function to fit the model to the training data
def regressorFit(stockList, regressor, trainset):
    for i in stockList:
        print("Fitting to", i)
        # Fit the model for each stock in the stockList
        history = regressor.fit(trainset[i]["X"], trainset[i]["y"], epochs=10, batch_size=200)
    return history 

# Function to initialize the LSTM model
def initModel(optimizer, loss):
    regressor = Sequential()
    # First LSTM layer with Dropout regularisation to prevent overfitting
    regressor.add(LSTM(units=128, return_sequences=True, input_shape=(pastDays, feature)))
    regressor.add(Dropout(0.2))
    # Second LSTM layer
    regressor.add(LSTM(units=64, return_sequences=True))
    regressor.add(Dropout(0.2))
    # Third LSTM layer
    regressor.add(LSTM(units=32, return_sequences=False))
    regressor.add(Dropout(0.2))
    # Output layer
    regressor.add(Dense(units=1, kernel_regularizer=regularizers.l2(0.01)))
    # Compile the model
    regressor.compile(optimizer, loss)
    return regressor

# Function to predict results and evaluate model performance
def predictResults(testScalers, stockName, testset, regressor, trainPercentage, df, pred_result):
    # Inverse transform the true values
    y_true = testScalers[stockName]["y"].inverse_transform(testset[stockName]["y"].reshape(-1, 1))
    # Make predictions using the regressor model
    prediction = regressor.predict(testset[stockName]["X"])

    # Calculate the split time between training and testing data
    time = (int)((len(df[stockName]) - 1) * trainPercentage)
    if (len(df[stockName]) - 1 - time) < pastDays:
        splitTime = df[stockName].index[len(df[stockName]) - 1 - pastDays]
    else:
        splitTime = df[stockName].index[time]
    
    # Scale the closing prices
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit_transform(df[stockName].loc[splitTime:, ["Close"]])

    # Inverse transform the predicted values
    y_pred = scaler.inverse_transform(prediction)
    
    # Calculate model evaluation metrics
    MABE = sklearn.metrics.mean_absolute_percentage_error(y_true[-pastDays * 2:], y_pred[-pastDays * 2:])
    MSE = sklearn.metrics.mean_squared_error(y_true[-pastDays * 2:], y_pred[-pastDays * 2:])
    
    # Store the results in pred_result dictionary
    pred_result[stockName] = {}
    pred_result[stockName]["True"] = y_true
    pred_result[stockName]["Pred"] = y_pred
    
    return pred_result, y_true, y_pred, MABE, MSE

# Function to print predictions and compare with true values
def printPredictions(y_pred, y_true, testset, stockName, df):
    success = 0
    fail = 0
    for j in range(0, len(y_pred)):
        # Calculate the split time for training and testing data
        trainPercentage = 0.7
        time = (int)((len(df[stockName]) - 1) * trainPercentage)
        if (len(df[stockName]) - 1 - time) < pastDays:
            splitTime = df[stockName].index[len(df[stockName]) - 1 - pastDays]
        else:
            splitTime = df[stockName].index[time]
        
        # Scale past closing prices
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit_transform(df[stockName].loc[splitTime:, ["Close"]])

        # Get the past closing prices, prediction, and true value
        pastX = np.array(df[stockName].loc[splitTime:, ["Close"]]) 
        past = pastX[j + pastDays - futureDays][0]
        pred = y_pred[j][0]
        true = y_true[j][0]
        
        # Check if the prediction matches the true value trend
        if ((pred - past) * (true - past)) > 0:
            print("GREAT SUCCESS")
            success += 1
        else:
            print("lowlife fail")
            fail += 1
        
        # Print prediction, true value, and past value
        print(f"Prediction= {pred}---True= {true}---Past= {past}")

    return success, fail

# Function to crop an image (used for chart cropping)
def cropImage(inputPath, outputPath, left, upper, right, lower):
    # Open the image
    image = Image.open(inputPath)
    
    # Crop the image
    cropped_image = image.crop((left, upper, right, lower))
    
    # Save the cropped image
    cropped_image.save(outputPath)

# Function to save predictions to a file
def writePredictionsToFile(pred_result, stockList):
    # Open a file for writing
    with open("predicted_stock_prices.txt", "w") as file:
        # Write headers
        file.write("Şirket Adı\tTahmin Edilen Fiyat\n")
        
        # Write the predicted prices for each stock
        for i in stockList:
            predicted_price = pred_result[i]["Pred"][-1][0]  # Last predicted price
            file.write(f"{i}\t{predicted_price}\n")

# Function to load the model if it exists, otherwise return None
def getModel():
    if(os.path.exists("Ağırlıklar/test.keras")):
        print("Model already exists")
        regressor = keras.models.load_model('Ağırlıklar/test.keras')
        return regressor
    else:
        return None

# Function to generate a list of stock names from CSV files
def csvtoStockList():
    stockList = []
    for file in os.listdir("./Veri Toplama/data/"):
        stockList.append(file.replace(".csv", ""))
    return stockList

# Function to plot and show graphs for predictions
def showGraph(stockName, y_true, y_pred, MABE, MSE):
    plt.figure(figsize=(14, 6))
    plt.title(f"{stockName} with MABE:{MABE} MSE:{MSE}")
    plt.plot(y_true[-pastDays * 2:], label="True Values")
    plt.plot(y_pred[-pastDays * 2:], ".", label="Predicted Values")
    plt.ylabel("Price")
    plt.xlabel("Date")
    plt.legend()
    plt.show()

# Function to save prediction graph as an image
def saveGraph(stockName, y_true, y_pred, MABE, MSE):
    plt.figure(figsize=(14, 6))
    plt.title(f"{stockName} with MABE:{MABE} MSE:{MSE}")
    plt.plot(y_true[-pastDays * 2:], label="True Values")
    plt.plot(y_pred[-pastDays * 2:], ".", label="Predicted Values")
    plt.ylabel("Price")
    plt.xlabel("Date")
    plt.legend()

    # Save the plot image
    plot_path = f"web/static/charts/{stockName}_prediction_plot.png"
    if not os.path.exists("web/static/charts/"):
        os.makedirs(plot_path)
    plt.savefig(plot_path)

    # Crop the image for web display
    cropped_plot_path = f"web/static/charts/{stockName}_prediction_plot.png"
    cropImage(plot_path, cropped_plot_path, 115, 40, 1400 - 125, 600 - 40) 
    plt.close()

# Function to split the data into training and testing sets
def splitTrainAndTest(df, stockList, trainPercentage):
    df_new = {}
    for i in stockList:
        df_new[i] = {}
        time = (int)((len(df[i]) - 1) * trainPercentage)
        if (len(df[i]) - 1 - time) < pastDays:
            splitTime = df[i].index[len(df[i]) - 1 - pastDays]
        else:
            splitTime = df[i].index[time]
        # Split data into training and testing
        df_new[i]["Train"] = df[i].loc[:splitTime, ["Close", "Volume"]]
        df_new[i]["Test"] = df[i].loc[splitTime:, ["Close", "Volume"]]
    return df_new

# Function to split data into features (X) and labels (y)
def splitDataXy(transform_train, transform_test, stockList, trainScalers, testScalers):
    testset = {}
    trainset = {}
    for j in stockList:
        testset[j] = {}
        X_test = []
        y_test = []
        trainset[j] = {}
        X_train = []
        y_train = []
        # Prepare X and y for testing data
        for i in range(pastDays, len(transform_test[j]) - futureDays + 1):
            X_test.append(transform_test[j][i - pastDays:i, [0, 1]])
            y_test.append(transform_test[j][i + futureDays - 1, 0])
        # Prepare X and y for training data
        for i in range(pastDays, len(transform_train[j]) - futureDays):
            X_train.append(transform_train[j][i - pastDays:i, [0, 1]])
            y_train.append(transform_train[j][i + futureDays, 0])

        # Convert lists to numpy arrays
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        y_test = testScalers[j]["y"].fit_transform(y_test.reshape(-1, 1))
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        y_train = trainScalers[j].fit_transform(y_train.reshape(-1, 1))

        # Scale X features
        X_shape = X_test.shape
        X_test = testScalers[j]["X"].fit_transform(X_test.reshape(-1, feature))
        X_test = np.array(X_test).reshape(X_shape)
        
        # Store the processed data
        testset[j]["X"] = X_test
        testset[j]["y"] = y_test 
        trainset[j]["X"] = X_train
        trainset[j]["y"] = y_train 
    return trainset, testset

# Function to scale the training and testing data
def scaleData(stockList, df_new):
    transform_train = {}
    trainScalers = {}
    transform_test = {}
    testScalers = {}
    for i in stockList:
        # Scale training data
        trainSc = MinMaxScaler(feature_range=(0, 1))
        a0 = df_new[i]["Train"].to_numpy()
        transform_train[i] = trainSc.fit_transform(a0)
        trainScalers[i] = trainSc

        # Scale testing data
        testScalers[i] = {}
        a1 = df_new[i]["Test"].to_numpy()
        transform_test[i] = a1
        testScalers[i]["X"] = MinMaxScaler(feature_range=(0, 1))
        testScalers[i]["y"] = MinMaxScaler(feature_range=(0, 1))

    return transform_train, trainScalers, transform_test, testScalers

# Function to read data from CSV files into DataFrames
def getDataFrames(stockList):
    df = {}
    for csv_file in stockList:
        df[csv_file] = pd.read_csv(f"./Veri Toplama/data/{csv_file}.csv", index_col="Date", parse_dates=True)
    return df
