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
from importlib import reload
import equity_classes
reload(equity_classes)
from equity_classes import classes as cl
from sklearn.model_selection import train_test_split


def reshape_dataset(data, timesteps):
	'''timesteps here should be set to 1 - for simplicity
	returns the input but in 3D form in equal spaced timesteps'''

	leftover = data.shape[0] % timesteps  # Reduce the data to a number divisible by 5
	data_sub = data[leftover:]
	data_sub = array(split(data_sub, len(data_sub) / timesteps))

	#If univariate input, returns reshaped from 2d to 3d - otherwise, returns 3d
	if data_sub.ndim == 2:
		return data_sub.reshape(data_sub.shape[0], data_sub.shape[1], 1)
	else:
		return data_sub


# convert history into inputs and outputs - includes previous day
def to_supervised(train, n_input, n_out):
	# flatten data
	data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
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
			y.append(data[in_end-1:out_end, 0]) #slightly different behavior here than the equivalent
			#classes function. This includes the previous day as well.
		# move along one time step
		in_start += 1
	return array(X), array(y)


def reshape_X_classical(data):
	return data.reshape(data.shape[0], data.shape[1] * data.shape[2])


def reshape_y_classical(data, n_out=5):
	'''This function takes the y output from to_supervised
	as well as the number of steps. It then calculates the
	% change for t against t-1. t0 looks at t(0-1). A cumulative
	% change is then calculated and 5 values are outputted as expected.'''
	pct_cume = []
	df = np.array(pd.DataFrame(data).pct_change(axis=1).iloc[:, 1:])

	for i in range(len(df)):
		pct_cume.append([sum(df[i, 0:x:1]) for x in range(0, n_out + 1)])

	return np.array(pct_cume)[:, 1:]


def create_scaler(train):
	s0, s1 = train.shape[0], train.shape[1]
	# train = train.reshape(s0 * s1, s2)
	scaler = RobustScaler()
	# scaler = MinMaxScaler(-1,1)
	scaler = scaler.fit(train)
	train = scaler.fit_transform(train)
	train_scale = train.reshape(s0, s1, 1)

	return train_scale, scaler


def apply_scaler(test, scaler):
	s0, s1 = test.shape[0], test.shape[1]
	# test = test.reshape(s0 * s1, s2)
	test = scaler.transform(test)
	test_scale = test.reshape(s0, s1, 1)

	return test_scale


# inverse scaling for a forecasted value
def invert_scale(self, scaler, yhat):
	inverted = scaler.inverse_transform(yhat)
	return inverted[0, :]

def process_data(ticker):

	headers = pd.read_csv (r'/home/ubuntu/stock_lstm/export_files/headers.csv')
	df = pd.read_csv (r'/home/ubuntu/stock_lstm/export_files/stock_history.csv', header=None, names=list(headers))
	df.index.name = 'date'

	df.reset_index(inplace=True)  # temporarily reset the index to get the week day for OHE
	df['date'] = pd.to_datetime(df['date'])
	df.drop_duplicates(['date', 'ticker', 'close'], inplace=True)
	df['day'] = list(map(lambda x: datetime.weekday(x), df['date']))  # adds the numeric day for OHE
	df.set_index('date', inplace=True)  # set the index back to the date field

	# use pd.concat to join the new columns with your original dataframe
	df = pd.concat([df,pd.get_dummies(df['day'],prefix='day',drop_first=True)],axis=1)

	df_close = df[df['ticker'] == ticker].sort_index(ascending=True)

	df_close = df_close.drop(['adj close', 'day', 'ticker'], axis=1)
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

df_reshape = reshape_dataset(np.array(dataset), 1)
X, y = to_supervised(df_reshape, 15, 5)

X_classical = pd.DataFrame(reshape_X_classical(X)) #Reshapes X into 1 row and all columns for the features
y_classical = reshape_y_classical(y, n_out=5) #Reshapes y to calculate % change

X_train, X_test, y_train, y_test = train_test_split(X_classical, y_classical, test_size=0.3, random_state=101)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

train_scale, scaler = create_scaler(X_train) #creates the scaler on train
test_scale = apply_scaler(X_test, scaler) #applies the scaler to test

y_label = []
for i in range(len(y_classical)):
	try:
		if y_classical[i, :].max() >= 0.05:
			y_label.append('Upweek')
		else:
			y_label.append('Shit week')
	break


