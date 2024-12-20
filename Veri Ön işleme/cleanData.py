from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os

# Normalize the Prices Data Using MinMaxScaler
def minmaxScale(prices):
    # Scale data to a range between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    prices_scaled = scaler.fit_transform(prices)
    return prices_scaled

# Format the CSV Files by Modifying Header and Removing Unnecessary Lines
def formatData(csv_files):
    try:
        for csv_file in csv_files:
            # Open each CSV file and read the lines
            with open(f"./Veri Toplama/data/{csv_file}", "r") as fr:
                lines = fr.readlines()                
                lines[0] = lines[0].replace("Price", "Date", 1)  # Update the header
                
                # Initialize pointer to manage which lines to write
                ptr = 1

                # Open in write mode to modify the file
                with open(f"./Veri Toplama/data/{csv_file}", "w") as fw:
                    for line in lines:
                        if ptr != 2 and ptr != 3:  # Skip unwanted lines (e.g., line 2 and 3)
                            fw.write(line)
                        ptr += 1

        print("Formatted")
    except:
        print("Formatting error")

# Remove Unnecessary Data and Files from CSV Files
def removingUselessData(csv_files):
    for csv_file in csv_files:
        # Load CSV file into a DataFrame
        dataFrame = pd.read_csv(f"./Veri Toplama/data/{csv_file}")
        
        # Extract only the 'Close' prices and keep necessary columns
        prices = dataFrame['Close'].values.reshape(-1, 1)
        columnsToKeep = ['Date','Close','Volume']
        dataFrame = dataFrame[columnsToKeep]
        
        # Remove specific files (based on predefined list) from the directory
        if(csv_file == "LMYK.csv" or csv_file == "ALTNY.csv" or csv_file == "LMKDC.csv" or csv_file == "OBAMS.csv"):
            os.remove(f"./Veri Toplama/data/{csv_file}")
            continue
        
        # Save the cleaned data back to CSV
        dataFrame.to_csv(f"./Veri Toplama/data/{csv_file}", index=False)

# Main Function to Process CSV Files
def main():
    csv_files = []

    # Gather all CSV files in the directory
    for file in os.listdir("./Veri Toplama/data/"):
        csv_files.append(file)
    
    # Format and clean the data files
    formatData(csv_files)
    removingUselessData(csv_files)


if __name__ == "__main__":
    main()