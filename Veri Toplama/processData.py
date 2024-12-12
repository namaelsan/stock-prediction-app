from bs4 import BeautifulSoup
import yfinance as yf
import requests
import pandas as pd
import os

def getTickers(url):
    # BIST100 listesindeki şirketlerin Tickerlarını indir
    page = requests.get(url).text
    page_bs = BeautifulSoup(page, "lxml")

    tickers = []
    column = page_bs.find("div", class_="column-type7 wmargin")
    ticker_elements =column.find_all("div", class_="comp-cell _02 vtable")

    for element in ticker_elements:
        ticker = element.get_text(strip=True).replace('\n', '') + ".IS"
        tickers.append(ticker)
    return tickers

def saveTickers(tickers):
    try:
        dataFrame = pd.DataFrame(tickers)
        dataFrame.columns = ["Ticker"]
        dataFrame.to_csv("Tickers.csv",index=False)
        print("Tickers saved to file")
    except:
        return False
    return True

def getData(tickers):
    # yfinance kullanarak bist100 listesindeki hisselerin son 5 yıldaki hisse değerlerini indir.

    for ticker in tickers:
        data = yf.download(ticker, period='5y')
        if data.empty:
            data = yf.download(ticker,period='max')
        data.to_csv(f"./Veri Toplama/data/{ticker.replace('.IS','')}.csv")
    
    return data

# def mergeData():
#     # getData ile elde edilen hisse değerlerini tek bir MergeData.csv dosyasına yaz
#     mergedData = pd.DataFrame()
#     csv_files = []

#     for file in os.listdir("./data/"):
#         csv_files.append(file)
    
#     for csv_file in csv_files:
#         data = pd.read_csv(os.path.join("./data/",csv_file))
#         data['Ticker'] = os.path.splitext(csv_file)[0]

    return True

if (__name__ == "__main__"):
    # getData ve mergeData kullanılarak bir csv dosyası oluştur.
    url = "https://www.kap.org.tr/en/Endeksler"
    tickers = getTickers(url)
    saveTickers(tickers)
    getData(tickers)

    exit()