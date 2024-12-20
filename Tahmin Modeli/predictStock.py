from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import lib

def main():

    stockList = lib.csvtoStockList()

    df = lib.getDataFrames(stockList)

    df_new = lib.splitTrainAndTest(df,stockList,0.7)

    transform_train, trainScalers, transform_test, testScalers = lib.scaleData(stockList,df_new)

    trainset,testset = lib.splitDataXy(transform_train,transform_test,stockList,trainScalers,testScalers)
    regressor = lib.getModel()
    
    if(regressor == None):
        print("no model found")
        exit()

    trainPercentage=0.7
    totalSuccess=0
    pred_result = {}

    for stockName in stockList:
        pred_result,y_true,y_pred,MABE,MSE = lib.predictResults(testScalers,stockName,testset,regressor,trainPercentage,df,pred_result)

        success,fail = lib.printPredictions(y_pred,y_true,testset,stockName,df)

        totalSuccess+=(success/(success+fail))
        print(f"Succes Rate:{success*100/(success+fail):.2f}")

        lib.showGraph(stockName,y_true,y_pred,MABE,MSE)
        lib.saveGraph(stockName,y_true,y_pred,MABE,MSE)

    # Tahmin sonuçlarını bir dosyaya yazdırıyoruz
    lib.writePredictionsToFile(pred_result, stockList)
    print(f"Overall Success:{totalSuccess*100/len(stockList)}")


if __name__ == "__main__":
    main()