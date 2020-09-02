# univariate multi-step cnn for the power usage dataset
from math import sqrt
from numpy import split, array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from sklearn.preprocessing import  StandardScaler, MinMaxScaler, RobustScaler
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime, date


class prepare_univariate_rnn:

    def __init__(self, ticker):

        self.ticker = ticker
        pass

    def reshape_dataset(self, data, timesteps):
        '''timesteps here should be set to 1 - for simplicity
        returns the input but in 3D form in equal spaced timesteps'''

        leftover = data.shape[0] % timesteps  # Reduce the data to a number divisible by 5
        data_sub = data[leftover:]
        data_sub = np.array(np.split(data_sub, len(data_sub) / timesteps))

        # If univariate input, returns reshaped from 2d to 3d - otherwise, returns 3d
        if data_sub.ndim == 2:
            return data_sub.reshape(data_sub.shape[0], data_sub.shape[1], 1)
        else:
            return data_sub

    # create a differenced series
    def get_difference(self, data, interval=1):
        '''Takes in a dataset and an interval (default is 1), then returns
        a differenced value'''
        s0, s1, s2 = data.shape[0], data.shape[1], data.shape[2]
        data = data.reshape(s0 * s1, s2)
        diff = list()
        for i in range(interval, len(data)):
            value = data[i] - data[i - interval]
            diff.append(value)
        #plt.plot(diff[1:])
        #plt.show()
        return diff

    def split_dataset(self, data, train_pct):

        '''performs the splitting of the already reshaped dataset'''

        train_weeks = int(data.shape[0] * train_pct)
        train, test = data[0:train_weeks, :], data[train_weeks:, :]

        return train, test

    # convert history into inputs and outputs
    def to_supervised(self, data_in, n_input, n_out):
        # flatten data
        data = data_in.reshape((data_in.shape[0] * data_in.shape[1], data_in.shape[2]))
        X, y = list(), list()
        in_start = 0
        # step over the entire history one time step at a time
        for _ in range(len(data)):
            # define the end of the input sequence
            in_end = in_start + n_input
            out_end = in_end + n_out
            # ensure we have enough data for this instance
            if out_end <= len(data):
                X.append(data[in_start:in_end, :])
                y.append(data[in_end:out_end, 0])
            # move along one time step
            in_start += 1
        return array(X), array(y)

    def create_scaler(self, train):
        s0, s1 = train.shape[0], train.shape[1]
        # train = train.reshape(s0 * s1, s2)
        scaler = RobustScaler()
        # scaler = MinMaxScaler(-1,1)
        scaler = scaler.fit(train)
        train = scaler.fit_transform(train)
        train_scale = train.reshape(s0, s1, 1)

        return train_scale, scaler

    def apply_scaler(self, test, scaler):

        s0, s1 = test.shape[0], test.shape[1]
        #test = test.reshape(s0 * s1, s2)
        test = scaler.transform(test)
        test_scale = test.reshape(s0, s1, 1)

        return test_scale

    # inverse scaling for a forecasted value
    def invert_scale(self, scaler, yhat):
        inverted = scaler.inverse_transform(yhat)
        return inverted[0, :]

    def forecast(self, model, history, n_input, scaler, interval):
        # flatten data
        data = array(history)
        data = self.apply_scaler(np.array(self.get_difference(data, interval)), scaler)  # difference the history data to prepare next test sample
        data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
        # retrieve last observations for input data
        input_x = data[-n_input:, 0]
        # reshape into [1, n_input, 1]
        input_x = input_x.reshape((1, len(input_x), 1))
        # forecast the next week on the differenced data
        yhat = model.predict(input_x, verbose=0)
        # yhat = invert_scale(scaler, yhat) #invert the scale back to differenced only
        # yhat = yhat[0]

        #return yhat
        return self.invert_scale(scaler, yhat)

    def process_data(self):

        headers = pd.read_csv(r'/home/ubuntu/stock_lstm/export_files/headers.csv')
        df = pd.read_csv(r'/home/ubuntu/stock_lstm/export_files/stock_history.csv', header=None, names=list(headers))
        df.index.name = 'date'

        df.reset_index(inplace=True)  # temporarily reset the index to get the week day for OHE
        df['date'] = pd.to_datetime(df['date'])
        df.drop_duplicates(['date', 'ticker', 'close'], inplace=True)
        df['day'] = list(map(lambda x: datetime.weekday(x), df['date']))  # adds the numeric day for OHE
        df.set_index('date', inplace=True)  # set the index back to the date field

        # use pd.concat to join the new columns with your original dataframe
        df = pd.concat([df, pd.get_dummies(df['day'], prefix='day', drop_first=True)], axis=1)

        df_close = df[df['ticker'] == self.ticker].sort_index(ascending=True)

        df_close = df_close.drop(['adj close', 'day', 'ticker',
                                  'volume_delta', 'prev_close_ch', 'prev_volume_ch', 'macds', 'macd', 'dma', 'macdh',
                                  'ma200'],
                                 axis=1)
        df_close = df_close.sort_index(ascending=True, axis=0)

        # Move the target variable to the end of the dataset so that it can be split into X and Y for Train and Test
        cols = list(df_close.columns.values)  # Make a list of all of the columns in the df
        cols.pop(cols.index('close'))  # Remove outcome from list
        df_close = df_close[['close'] + cols]  # Create new dataframe with columns in correct order

        df_close = df_close.dropna()

        fig, axes = plt.subplots(figsize=(16, 8))
        # Define the date format
        axes.xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # to display ticks every 3 months
        axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # to set how dates are displayed
        axes.set_title(self.ticker)
        axes.plot(df_close.index, df_close['close'], linewidth=3)
        plt.show()

        return df_close
