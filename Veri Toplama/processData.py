from bs4 import BeautifulSoup
import yfinance as yf
import requests
import pandas as pd
import os

# Get the Tickers from the BIST100 List
def getTickers(url):
    # Fetch the page content and parse it with BeautifulSoup
    page = requests.get(url).text
    page_bs = BeautifulSoup(page, "lxml")

    tickers = []
    column = page_bs.find("div", class_="column-type7 wmargin")
    ticker_elements = column.find_all("div", class_="comp-cell _02 vtable")

    # Extract tickers and append ".IS" to indicate Borsa Istanbul
    for element in ticker_elements:
        ticker = element.get_text(strip=True).replace('\n', '') + ".IS"
        tickers.append(ticker)
    return tickers

# Save the Tickers to a CSV File
def saveTickers(tickers):
    try:
        # Create a DataFrame from the tickers list and save it as a CSV
        dataFrame = pd.DataFrame(tickers)
        dataFrame.columns = ["Ticker"]
        dataFrame.to_csv("Tickers.csv", index=False)
        print("Tickers saved to file")
    except:
        return False
    return True

# Download Stock Data for the Last 5 Years Using yfinance
def getData(tickers):
    # For each ticker, download stock data for the last 5 years
    for ticker in tickers:
        data = yf.download(ticker, period='5y')
        
        # If no data found, fetch the maximum available data
        if data.empty:
            data = yf.download(ticker, period='max')
        
        # Save the stock data as a CSV file
        data.to_csv(f"./Veri Toplama/data/{ticker.replace('.IS', '')}.csv")
    
    return data

# def mergeData():
#     # Merge all stock data into a single CSV file
#     mergedData = pd.DataFrame()
#     csv_files = []

#     for file in os.listdir("./data/"):
#         csv_files.append(file)
    
#     for csv_file in csv_files:
#         data = pd.read_csv(os.path.join("./data/",csv_file))
#         data['Ticker'] = os.path.splitext(csv_file)[0]

    return True

# Main function: Fetch tickers, save them, and download stock data
def main():
    url = "https://www.kap.org.tr/en/Endeksler"
    tickers = getTickers(url)
    saveTickers(tickers)
    getData(tickers)

    exit()


if (__name__ == "__main__"):
    main()