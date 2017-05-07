import os
import datetime
import pandas as pd
import numpy as np

def get_data_csv(symbol, dates, data_dir="data/", volume=True, redownload=False):

    #create directory if it does not exist.
    if not os.path.exists(data_dir):
      os.makedirs(data_dir)

    file_full_path = data_dir + symbol + '.csv'

    # if the file does not exist or redownload is True, then download the data file from the web.
    if not os.path.isfile(file_full_path) or redownload:
        url = "http://ichart.finance.yahoo.com/table.csv?s=" + symbol
        data = pd.read_csv(url)
        data.to_csv(file_full_path)
    
    # after data is downloaded, if the data file does not exist in the folder, error out.
    if not os.path.isfile(file_full_path):
        print('No data for ' + symbol)
        exit(0)

    # create a data frame with dates as index
    df = pd.DataFrame(index=dates)

    # column names for the dataframe
    cols = ['Date', 'Open', 'Close', 'Low', 'High', 'Adj Close']
    if volume:
        cols.append('Volume')

    # read the csv file into pandas dataframe and return it.
    data = pd.read_csv(file_full_path, parse_dates = True, index_col='Date', usecols=cols).astype(np.float32)
    df = df.join(data)
    return df


