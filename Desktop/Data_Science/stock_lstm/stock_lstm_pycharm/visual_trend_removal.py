import matplotlib
from keras.utils import plot_model

matplotlib.get_backend ()
import pandas as pd
import numpy as np
#import pandas_datareader as pdr
#from numpy.core._multiarray_umath import ndarray
from matplotlib import pyplot as plt
import scipy.stats as stats
import seaborn as sns
from datetime import datetime
sns.set (style="darkgrid", color_codes=True)
from pandas import read_csv
from math import sqrt
from numpy import split, array
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler
import matplotlib.dates as mdates
import os
from sklearn.metrics import classification_report



# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return pd.Series(diff)

def difference_pct(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = (dataset[i] - dataset[i - interval]) / dataset[i - interval]
		diff.append(value)
	return pd.Series(diff)

def bool_change(df, col):
    df = difference(df[col].reset_index(drop=True))
    return df.apply(lambda x: 1 if x > 0 else 0)

def timeseries_to_supervised(data, timesteps, predsteps):
    x_data, y_data = [ ], [ ]
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range (len (data)):
        # define the end of the input sequence
        in_end = in_start + timesteps
        out_end = in_end + predsteps
        # ensure there is enough to make a 5 day prediction
        if out_end <= len (data):
            x_data.append (data [ in_start:in_end, : ])
            y_data.append (data [ in_end:out_end, 0 ]) #[:, 0] might have to factor
        # move along one time step
        in_start += 1
    x_data, y_data = np.array (x_data), np.array (y_data)
    return x_data, y_data


def split_data(X, y):
    '''Takes in a dataset and returns an 80/20 split of train
    and test in the form of a numpy array'''

    split_size = int((int(X.shape[0]) * train_pct))
    x_train, x_dev = X[0:split_size, :, :], X[split_size:, :, :]
    y_train, y_dev = y[ 0:split_size, :], y [ split_size:, :]

    assert int(x_train.shape[0]) + int(x_dev.shape[0]) == int(X.shape[0])
    assert int(y_train.shape [ 0 ]) + int (y_dev.shape [ 0 ]) == int (y.shape [ 0 ])

    return x_train, y_train, x_dev, y_dev

def unshape_supervised(X, y):
    '''Unshape the train and dev from their shaped form, in order to scale'''
    X = X.reshape(X.shape[0] * X.shape[1], X.shape[2])
    y = y.reshape(y.shape[0] * y.shape[1], 1)
    return X, y

def scale_data(x_train, y_train, x_dev, y_dev):

    '''We have to unshape the data as a first step for both train and dev,
    then apply the scaler to both X and y for both, then reshape the scaled
    data back to the original scale.'''

    X_t, y_t = unshape_supervised (x_train, y_train)
    X_d, y_d = unshape_supervised (x_dev, y_dev)

    X_scaler = MinMaxScaler (feature_range=(0, 1))
    X_scaler = X_scaler.fit (X_t)
    y_scaler = MinMaxScaler (feature_range=(0, 1))
    y_scaler = y_scaler.fit (y_t)

    X_train_norm = X_scaler.fit_transform (X_t).reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2])
    y_train_norm = y_scaler.fit_transform (y_t).reshape(y_train.shape[0], y_train.shape[1])

    X_dev_norm = X_scaler.fit_transform (X_d).reshape(x_dev.shape[0], x_dev.shape[1], x_dev.shape[2])
    y_dev_norm = y_scaler.fit_transform (y_d).reshape(y_dev.shape[0], y_dev.shape[1])

    assert X_train_norm.shape == x_train.shape
    assert y_train_norm.shape == y_train.shape
    assert X_dev_norm.shape == x_dev.shape
    assert y_dev_norm.shape == y_dev.shape

    return X_train_norm, y_train_norm, X_dev_norm, y_dev_norm, X_scaler, y_scaler


# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# inverse scaling for a forecasted value
def invert_scale(scaler, yhat):
    inverted = scaler.inverse_transform (yhat)
    return inverted[0, :]

def chart_results(predsteps, predictions, y_dev, plot_days, y_dev_raw ):
    days = [ "Day" + str (i) for i in range (1, predsteps + 1) ]
    plot_days = [ "Day" + str (i) for i in range (1, plot_days + 1) ]
    df_pred = pd.DataFrame.from_records(np.row_stack(predictions), columns=[ i for i in days ]).stack ().reset_index ()
    #df_actual = pd.DataFrame.from_records(np.row_stack (y_dev_raw,columns=[ i for i in days ]).stack ().reset_index ()
    #df_pred.rename (columns={'level_0': 'iteration', 'level_1': 'day', 0: 'pred'}, inplace=True)
    df_actual = pd.DataFrame.from_records (np.row_stack(y_dev), columns=[ i for i in days ]).stack ().reset_index ()
    df_pred.rename (columns={'level_0': 'iteration', 'level_1': 'day', 0: 'pred'}, inplace=True)

    df_pred [ 'actual' ] = df_actual [ 0 ]
    df_pred = df_pred [ df_pred [ 'day' ].isin (plot_days) ]
    df_pred.reset_index(inplace=True)

    fig, axes = plt.subplots (figsize=(16, 8))

    axes.plot (df_pred[ 'pred' ], label='predicted', linewidth=1)
    axes.plot (df_pred[ 'actual' ], label='actual', linewidth=1)
    axes.plot(y_dev_raw[:, :1], label='dev actual', linewidth=1)

    #axes.set_title (str (i))

    axes.legend ([ 'Pred', 'Actual', 'GT Actual' ], loc='upper right')
    plt.show ()

    return df_pred


# execute the experiment
def process_data():
    #project_path = '/Users/jamesm/Desktop/Data_Science/stock_nn'
    project_path = r'/home/ubuntu/tmp/pycharm_project_162'
    os.chdir (project_path)

    # Import the Dataset
    df = pd.read_csv ('aapl.csv', low_memory=False)
    df [ 'date' ] = pd.to_datetime (df.date)
    df.index = df [ 'date' ]

    # df.dropna (inplace=True)
    ticker = str (df [ 'ticker' ].unique () [ 1:-1 ])

    # Create figure and plot space
    fig, axes = plt.subplots (figsize=(16, 8))
    # Define the date format
    axes.xaxis.set_major_locator (mdates.MonthLocator (interval=6))  # to display ticks every 3 months
    axes.xaxis.set_major_formatter (mdates.DateFormatter ('%Y-%m'))  # to set how dates are displayed
    axes.set_title (ticker)
    axes.plot (df.index, df [ 'close' ], linewidth=3)
    plt.show ()

    df_close = df.drop ([ 'adj close', 'date', 'ticker',
    'volume_delta', 'prev_close_ch', 'prev_volume_ch', 'macds', 'macd', 'dma', 'macdh', 'ma200' ], axis=1)
    df_close = df_close.sort_index (ascending=True, axis=0)

#Move the target variable to the end of the dataset so that it can be split into X and Y for Train and Test
    cols = list(df_close.columns.values) #Make a list of all of the columns in the df
    cols.pop(cols.index('close')) #Remove outcome from list
    df_close = df_close[['close']+cols] #Create new dataframe with columns in correct order

    df_close = df_close.dropna ()

    return df_close

repeats = 1
timesteps = 10
predsteps = 5
train_pct = 0.8

df_close = process_data()


#High level correlation
fig, axes = plt.subplots(nrows=1, figsize=(15,8))
sns.heatmap(df_close.corr(), cmap='coolwarm', annot=True)
plt.show()

#Inter Variable Correlations
for num in range(len(df_close.columns)):
    fig, axes = plt.subplots (figsize=(6, 6))
    df_close.corr().iloc[:, num].sort_values(ascending=False).iloc[:-1,].plot.bar()
    plt.title (df_close.columns [ num])
    plt.show()


#Now consider detrending using the previous day
diff_values = df_close.apply(difference, args=[1])

X = diff_values.iloc[:, 1:]
y = diff_values.iloc[:, 0]

train_pct = 0.8
split_size = int((int(X.shape[0]) * train_pct))
x_train, x_dev = X.iloc[0:split_size], X.iloc[split_size:]
y_train, y_dev = y.iloc[ 0:split_size], y.iloc[ split_size:]

fig, axes = plt.subplots (figsize=(6, 6))
y_train.plot.line (lw=2)
y_dev.plot.line (lw=2)
plt.show()

for num in range(x_train.shape[1]):
    fig, axes = plt.subplots (figsize=(6, 6))
    x_train.iloc [ :, num ].plot.line (lw=2)
    x_dev.iloc [ :, num ].plot.line (lw=2)
    plt.title (df_close.columns [num+1])
    plt.show()

#Now consider detrending using % difference from previous day
diff_pct = df_close.pct_change()
diff_pct2 = df_close.apply(difference_pct, args=[1])

X = diff_pct.iloc[:, 1:]
y = diff_pct.iloc[:, 0]

train_pct = 0.8
split_size = int((int(X.shape[0]) * train_pct))
x_train, x_dev = X.iloc[0:split_size], X.iloc[split_size:]
y_train, y_dev = y.iloc[ 0:split_size], y.iloc[ split_size:]

fig, axes = plt.subplots (figsize=(6, 6))
y_train.plot.line (lw=2)
y_dev.plot.line (lw=2)
plt.show()

for num in range(x_train.shape[1]):
    fig, axes = plt.subplots (figsize=(6, 6))
    x_train.iloc [ :, num ].plot.line (lw=2)
    x_dev.iloc [ :, num ].plot.line (lw=2)
    plt.title (df_close.columns [num+1])
    plt.show()

#Now look at the distplots for train and test for the comparable fields
for num in range(x_train.shape[1]):
    fig, axes = plt.subplots (figsize=(6, 6))
    sns.distplot(x_train.iloc [ 1:, num ])
    sns.distplot(x_dev.iloc [ 1:, num ])
    plt.title (df_close.columns [ num + 1 ])
plt.show()

fig, axes = plt.subplots (figsize=(6, 6))
sns.distplot(y_train.iloc [ 1:])
sns.distplot(y_dev.iloc [ 1:])
plt.title (df_close.columns [ 0 ])
plt.show()


#Now look at the dist plots for df_close
df = df_close

X = df.iloc[:, 1:]
y = df.iloc[:, 0]

train_pct = 0.8
split_size = int((int(X.shape[0]) * train_pct))
x_train, x_dev = X.iloc[0:split_size], X.iloc[split_size:]
y_train, y_dev = y.iloc[ 0:split_size], y.iloc[ split_size:]


#Now look at the distplots for train and test for the comparable fields
for num in range(x_train.shape[1]):
    fig, axes = plt.subplots (figsize=(6, 6))
    sns.distplot(x_train.iloc [ 1:, num ])
    sns.distplot(x_dev.iloc [ 1:, num ])
    plt.title (df_close.columns [ num + 1 ])
plt.show()

fig, axes = plt.subplots (figsize=(6, 6))
sns.distplot(y_train.iloc [ 1:])
sns.distplot(y_dev.iloc [ 1:])
plt.title (df_close.columns [ 0 ])
plt.show()

#Now look at the dist plots for df_close
diff_values = df_close.apply(difference, args=[1])

X = diff_values.iloc[:, 1:]
y = diff_values.iloc[:, 0]

train_pct = 0.8
split_size = int((int(X.shape[0]) * train_pct))
x_train, x_dev = X.iloc[0:split_size], X.iloc[split_size:]
y_train, y_dev = y.iloc[ 0:split_size], y.iloc[ split_size:]


#Now look at the distplots for train and test for the comparable fields
for num in range(x_train.shape[1]):
    fig, axes = plt.subplots (figsize=(6, 6))
    sns.distplot(x_train.iloc [ 1:, num ])
    sns.distplot(x_dev.iloc [ 1:, num ])
    plt.title (df_close.columns [ num + 1 ])
plt.show()

fig, axes = plt.subplots (figsize=(6, 6))
sns.distplot(y_train.iloc [ 1:])
sns.distplot(y_dev.iloc [ 1:])
plt.title (df_close.columns [ 0 ])
plt.show()

