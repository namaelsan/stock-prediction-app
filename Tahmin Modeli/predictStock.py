from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
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
    totalPercentage=0
    pred_result = {}

    for stockName in stockList:
        y_true = testScalers[stockName]["y"].inverse_transform(testset[stockName]["y"].reshape(-1,1))
        prediction=regressor.predict(testset[stockName]["X"])

        time = (int) ((len(df[stockName]) -1) * trainPercentage)
        if (len(df[stockName])-1 - time) < lib.pastDays:
            splitTime= df[stockName].index[len(df[stockName])-1 -lib.pastDays]
        else:
            splitTime= df[stockName].index[time]
        scaler = None
        scaler = MinMaxScaler(feature_range=(0,1))
        scaler.fit_transform(df[stockName].loc[splitTime:,["Close"]])

        y_pred = scaler.inverse_transform(prediction)
        MABE = mean_absolute_percentage_error(y_true, y_pred)
        pred_result[stockName] = {}
        pred_result[stockName]["True"] = y_true
        pred_result[stockName]["Pred"] = y_pred
        
        success=0
        fail=0

        for j in range(0,len(y_pred)):

            pastX=np.stack(testset[stockName]["X"])
            shape1 =(pastX.shape[0],pastX.shape[1],lib.feature)
            pastXShaped = testScalers[stockName]["X"].inverse_transform(pastX.reshape(-1,lib.feature))
            pastTransformed = np.reshape(pastXShaped, shape1)


            past=pastTransformed[j][lib.pastDays - lib.futureDays][0]
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
            b=testset[stockName]["X"][j][0][0]


        print(f"Succes Rate:{(success/(success+fail)):.2f}")
        totalPercentage+=success/(success+fail)

        lib.showGraph(stockName,y_true,y_pred,MABE)
        lib.saveGraph(stockName,y_true,y_pred,MABE)

    # Tahmin sonuçlarını bir dosyaya yazdırıyoruz
    lib.writePredictionsToFile(pred_result, stockList)
    print(f"Overall Success:{totalPercentage/len(stockList)}")


if __name__ == "__main__":
    main()