import lib
from keras.optimizers import Adam

def main():

    stockList = lib.csvtoStockList()

    df = lib.getDataFrames(stockList)

    df_new = lib.splitTrainAndTest(df,stockList,trainPercentage=0.7)

    transform_train, trainScalers, transform_test, testScalers = lib.scaleData(stockList,df_new)

    del df_new

    trainset,testset = lib.splitDataXy(transform_train,transform_test,stockList,trainScalers,testScalers)

    regressor = lib.initModel(Adam(learning_rate=0.0005),'huber')

    lossHistory = lib.regressorFit(stockList,regressor,trainset)

    regressor.save("Ağırlıklar/test.keras")


if __name__ == "__main__":
    futureDays=1
    pastDays=30
    feature=2
    main()