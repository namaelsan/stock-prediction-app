import lib
from keras.optimizers import Adam

def main():

    stockList = lib.csvtoStockList()

    df = lib.getDataFrames(stockList)

    df_new = lib.splitTrainAndTest(df,stockList,trainPercentage=0.7)

    transform_train, trainScalers, transform_test, testScalers = lib.scaleData(stockList,df_new)

    for stockName in stockList:
        lib.showTrainTestGraph(stockName,df_new[stockName]["Train"],df_new[stockName]["Test"])

    del df_new

    trainset,testset = lib.splitDataXy(transform_train,transform_test,stockList,trainScalers,testScalers)

    regressor = lib.initModel(Adam(learning_rate=0.001),'mse')

    lossHistory = lib.regressorFit(stockList,regressor,trainset)

    regressor.save("Ağırlıklar/test.keras")


if __name__ == "__main__":
    main()