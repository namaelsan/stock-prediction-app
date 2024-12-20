from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import lib

def main():
    # Get a list of stock names from CSV files
    stockList = lib.csvtoStockList()

    # Load the data for each stock into a dictionary of DataFrames
    df = lib.getDataFrames(stockList)

    # Split the data into training and testing sets (70% for training, 30% for testing)
    df_new = lib.splitTrainAndTest(df, stockList, 0.7)

    # Scale the training and testing data using MinMaxScaler
    transform_train, trainScalers, transform_test, testScalers = lib.scaleData(stockList, df_new)

    # Prepare the training and testing datasets with appropriate features (X) and labels (y)
    trainset, testset = lib.splitDataXy(transform_train, transform_test, stockList, trainScalers, testScalers)
    
    # Load the pre-trained model, if available
    regressor = lib.getModel()

    # If no model is found, print an error and exit
    if regressor is None:
        print("no model found")
        exit()

    # Initialize a variable to track the total success rate
    trainPercentage = 0.7
    totalSuccess = 0
    pred_result = {}

    # Iterate over each stock in the list to make predictions and evaluate performance
    for stockName in stockList:
        # Predict results using the regressor and calculate MABE (Mean Absolute Bias Error) and MSE (Mean Squared Error)
        pred_result, y_true, y_pred, MABE, MSE = lib.predictResults(testScalers, stockName, testset, regressor, trainPercentage, df, pred_result)

        # Print the success/failure rate for the current stock's predictions
        success, fail = lib.printPredictions(y_pred, y_true, testset, stockName, df)

        # Update the total success rate
        totalSuccess += (success / (success + fail))
        print(f"Success Rate: {success * 100 / (success + fail):.2f}%")

        # Display and save graphs comparing true vs. predicted values
        lib.showGraph(stockName, y_true, y_pred, MABE, MSE)
        lib.saveGraph(stockName, y_true, y_pred, MABE, MSE)

    # Write the prediction results for all stocks to a text file
    lib.writePredictionsToFile(pred_result, stockList)

    # Print the overall success rate across all stocks
    print(f"Overall Success: {totalSuccess * 100 / len(stockList):.2f}%")

# Run the main function if this script is executed directly
if __name__ == "__main__":
    main()
