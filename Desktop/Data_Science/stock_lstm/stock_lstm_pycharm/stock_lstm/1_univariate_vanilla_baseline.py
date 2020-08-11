import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
from yahoo_fin import stock_info
# from yahoo_fin.options import *
import stockstats
from stockstats import StockDataFrame as sdf
from pandas_datareader import data as pdr
import requests_html
from numpy import mean
from numpy import median

headers = pd.read_csv (r'/Users/jamesm/Desktop/Data_Science/stock_lstm/export_files/headers.csv')
df = pd.read_csv (r'/Users/jamesm/Desktop/Data_Science/stock_lstm/export_files/stock_history.csv', header=None, names=list(headers))

# Extract the close for 'AAPL' only
history = df[df['ticker'] == 'AAPL']['close'].sort_index(ascending=True)
history.index.name = 'date'

# one-step naive forecast
def naive_forecast(history, n):
    return history[-n]# test naive forecast


# one-step simple forecast
def simple_forecast(history, config):
    n, offset, avg_type = config
# persist value, ignore other config
    if avg_type == 'persist':
        return history[-n]
# collect values to average
    values = list()
    if offset == 1:
        values = history[-n:]
    else:
    # skip bad configs
        if n*offset > len(history):
            raise Exception('Config beyond end of data: %d %d' % (n,offset))
    # try and collect n values using offset
    for i in range(1, n+1):
        ix = i * offset
        values.append(history[-ix])
  # check if we can average
    if len(values) < 2:
        raise Exception('Cannot calculate average')
  # mean of last n values
    if avg_type == 'mean':
        return mean(values)
  # median of last n values
    return median(values)

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:]

# root mean squared error or rmse
def measure_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))

# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
    predictions = list()
# split dataset
    train, test = train_test_split(data, n_test) # seed history with training dataset
    history = [x for x in train]
  # step over each time step in the test set
    for i in range(len(test)):
    # fit model and make forecast for history
        yhat = simple_forecast(history, cfg)
    # store forecast in list of predictions
        predictions.append(yhat)
    # add actual observation to history for the next loop
        history.append(test[i])
  # estimate prediction error
    error = measure_rmse(test, predictions)
    return error