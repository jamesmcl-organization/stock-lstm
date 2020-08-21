# naive forecast strategies for the power usage dataset
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime, date

# split a univariate dataset into train/test sets - everything divides evenly into the number
# of timesteps. This will allow for weekly predictions. This function has to be rebuilt to allow
# for overlapping weeks
#def split_dataset(data, timesteps, train_pct):

#	leftover = data.shape[0]%timesteps 	# Reduce the data to a number divisible by 5
#	weeks = data // timesteps			# determine total number of trading weeks

#	data_sub = data[leftover:]			# Reduce the initial data to a number divisible by 5
#	train_weeks = int(((data_sub.shape[0] * train_pct) // timesteps) * timesteps)
#	train, test = data_sub[0:train_weeks], data_sub[train_weeks:]

#	train = array(split(train, len(train) / timesteps))
#	test = array(split(test, len(test) / timesteps))

#	return train, test

#train, test = 	split_dataset(dataset.values, 5, 0.8)
#print(train.shape)
#print(test.shape)

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
	#history = pd.concat(x_data, y_data)
	return x_data, y_data
	#return history

def split_data(X, y, train_pct):
    '''Takes in a dataset and returns an 80/20 split of train
    and test in the form of a numpy array'''

    split_size = int((int(X.shape[0]) * train_pct))
    x_train, x_dev = X[0:split_size, :, :], X[split_size:, :, :]
    y_train, y_dev = y[ 0:split_size, :], y [ split_size:, :]

    assert int(x_train.shape[0]) + int(x_dev.shape[0]) == int(X.shape[0])
    assert int(y_train.shape [ 0 ]) + int (y_dev.shape [ 0 ]) == int (y.shape [ 0 ])

    #return x_train, y_train, x_dev, y_dev
	return x_train, x_dev



# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
	scores = list()
	# calculate an RMSE score for each day
	for i in range(actual.shape[1]):
		# calculate mse
		mse = mean_squared_error(actual[:, i], predicted[:, i])
		# calculate rmse
		rmse = sqrt(mse)
		# store
		scores.append(rmse)
	# calculate overall RMSE
	s = 0
	for row in range(actual.shape[0]):
		for col in range(actual.shape[1]):
			s += (actual[row, col] - predicted[row, col])**2
	score = sqrt(s / (actual.shape[0] * actual.shape[1]))
	return score, scores

# summarize scores
def summarize_scores(name, score, scores):
	s_scores = ', '.join(['%.1f' % s for s in scores])
	print('%s: [%.3f] %s' % (name, score, s_scores))

# evaluate a single model
def evaluate_model(model_func, train, test):
	# history is a list of weekly data
	history = [x for x in train]
	# walk-forward validation over each week
	predictions = list()
	for i in range(len(test)):
		# predict the week
		yhat_sequence = model_func(history)
		# store the predictions
		predictions.append(yhat_sequence)
		# get real observation and add to history for predicting the next week
		history.append(test[i, :])
	predictions = array(predictions)
	# evaluate predictions days for each week
	score, scores = evaluate_forecasts(test[:, :, 0], predictions)
	return score, scores

# daily persistence model
def daily_persistence(history):
	# get the data for the prior week
	last_week = history[-1]
	# get the total active power for the last day
	value = last_week[-1, 0]
	# prepare 7 day forecast
	forecast = [value for _ in range(5)]
	return forecast

# weekly persistence model
def weekly_persistence(history):
	# get the data for the prior week
	last_week = history[-1]
	return last_week[:, 0]

# week one year ago persistence model
#def week_one_year_ago_persistence(history):
	# get the data for the prior week
#	last_week = history[-52]
#	return last_week[:, 0]

def process_data(ticker):

	headers = pd.read_csv (r'/home/ubuntu/stock_lstm/export_files/headers.csv')
	df = pd.read_csv (r'/home/ubuntu/stock_lstm/export_files/stock_history.csv', header=None, names=list(headers))
	df.index.name = 'date'

	df.reset_index (inplace=True)		#temporarily reset the index to get the week day for OHE
	df['date']= pd.to_datetime(df['date'])
	df [ 'day' ] = list (map (lambda x: datetime.weekday(x), df [ 'date' ])) #adds the numeric day for OHE
	df.set_index('date', inplace=True) #set the index back to the date field

	# use pd.concat to join the new columns with your original dataframe
	df = pd.concat([df,pd.get_dummies(df['day'],prefix='day',drop_first=True)],axis=1)

	df_close = df[df['ticker'] == ticker].sort_index(ascending=True)

	df_close = df_close.drop(['adj close', 'day', 'ticker',
						'volume_delta', 'prev_close_ch', 'prev_volume_ch', 'macds', 'macd', 'dma', 'macdh', 'ma200'],
					   axis=1)
	df_close = df_close.sort_index(ascending=True, axis=0)

	# Move the target variable to the end of the dataset so that it can be split into X and Y for Train and Test
	cols = list(df_close.columns.values)  # Make a list of all of the columns in the df
	cols.pop(cols.index('close'))  # Remove outcome from list
	df_close = df_close[['close'] + cols]  # Create new dataframe with columns in correct order

	df_close = df_close.dropna()

	fig, axes = plt.subplots (figsize=(16, 8))
		# Define the date format
	axes.xaxis.set_major_locator (mdates.MonthLocator (interval=6))  # to display ticks every 3 months
	axes.xaxis.set_major_formatter (mdates.DateFormatter ('%Y-%m'))  # to set how dates are displayed
	axes.set_title (ticker)
	axes.plot (df_close.index, df_close [ 'close' ], linewidth=3)
	plt.show ()

	return df_close

dataset = process_data('AAPL')

# split into train and test

X, y = timeseries_to_supervised(dataset.values, 5, 5)
train, test = split_data(X, y, 0.8)
# define the names and functions for the models we wish to evaluate
models = dict()
models['daily'] = daily_persistence
models['weekly'] = weekly_persistence
#models['week-oya'] = week_one_year_ago_persistence
# evaluate each model
days = ['mon', 'tue', 'wed', 'thr', 'fri']
for name, func in models.items():
	# evaluate and get scores
	score, scores = evaluate_model(func, train, test)
	# summarize scores
	summarize_scores(name, score, scores)
	# plot scores
	plt.plot(days, scores, marker='o', label=name)
# show plot
plt.legend()
plt.show()