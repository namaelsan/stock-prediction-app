import lib
from keras.optimizers import Adam

def main():
    # Get the list of stock names from CSV files
    stockList = lib.csvtoStockList()

    # Load the data for each stock into a dictionary of DataFrames
    df = lib.getDataFrames(stockList)

    # Split the data into training and testing sets based on a specified percentage (70% for training, 30% for testing)
    df_new = lib.splitTrainAndTest(df, stockList, trainPercentage=0.7)

    # Scale the training and testing data using MinMaxScaler
    transform_train, trainScalers, transform_test, testScalers = lib.scaleData(stockList, df_new)

    # Clean up by deleting the original training and testing data after scaling
    del df_new

    # Prepare the training and testing datasets with appropriate X (features) and y (targets) values
    trainset, testset = lib.splitDataXy(transform_train, transform_test, stockList, trainScalers, testScalers)

    # Initialize the model with the Adam optimizer and mean squared error (MSE) loss function
    regressor = lib.initModel(Adam(learning_rate=0.001), 'mse')

    # Fit the model using the training data and stock list, and capture the loss history
    lossHistory = lib.regressorFit(stockList, regressor, trainset)

    # Save the trained model to disk
    regressor.save("Ağırlıklar/test.keras")

# Run the main function if this script is executed directly
if __name__ == "__main__":
    main()
