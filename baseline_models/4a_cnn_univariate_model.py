# univariate multi-step cnn for the power usage dataset
from math import sqrt
from numpy import split, array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime, date

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

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		#value = dataset[i] - dataset[i - interval] / dataset[i - interval] # % differencing
		diff.append(value)
	#plt.plot(diff[1:])
	#plt.show()
	return diff

def split_dataset(data, train_pct):

	'''performs the splitting of the already reshaped dataset'''

	train_weeks = int(data.shape[0] * train_pct)
	train, test = data[0:train_weeks, :], data[train_weeks:, :]

	return train, test

# convert history into inputs and outputs
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
			y.append(data[in_end:out_end, 0])
		# move along one time step
		in_start += 1
	return array(X), array(y)

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


	# train the model
def build_model(train, n_input, n_out):

	# prepare data - use differenced data at the point of training
	train_x, train_y = to_supervised(np.array(difference(train, interval=1)), n_input, n_out)

	# define parameters
	verbose, epochs, batch_size = 1, 1000, 16
	n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
	# define model
	model = Sequential()
	model.add(Conv1D(16, 3, activation='tanh', input_shape=(n_timesteps,n_features)))
	model.add(MaxPooling1D())
	model.add(Flatten())
	model.add(Dense(10, activation='tanh'))
	model.add(Dense(n_outputs))
	model.compile(loss='mse', optimizer='adam')
	# fit network
	model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
	return model

	# make a forecast
def forecast(model, history, n_input):
	# flatten data
	data = array(history)
	data = np.array(difference(data, interval=1)) #difference the history data to prepare next test sample
	data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
	# retrieve last observations for input data
	input_x = data[-n_input:, 0]
	# reshape into [1, n_input, 1]
	input_x = input_x.reshape((1, len(input_x), 1))
	# forecast the next week on the differenced data
	yhat = model.predict(input_x, verbose=0)
	yhat = yhat[0]
	return yhat

	# evaluate a single model
def evaluate_model(df, n_input, n_out):

	train, test = split_dataset(df, 0.8)
	model = build_model(train, n_input, n_out)
	history = [x for x in train]

	# walk-forward validation over each week
	predictions, actuals, predictions_ff = list(), list(), list()
	#predictions, actuals, predictions_ff, temp_yhat, temp_nstart = list(), list(), list(), list(), list()
	#predictions_diff, act_inv = list(), list()
	n_start = n_input
	for i in range(len(test)):
		# predict the week
		yhat_sequence = forecast(model, history, n_input)

		# store the predictions
		##predictions_diff.append(yhat_sequence) #can remove
		# get real observation and add to history for predicting the next n days using the forecast() function
		history.append(test[i, :])

		'''predictions is pulled from the previous, undifferenced 5 test actuals, then added to the
		differenced yhat sequence: predictions.append(test[n_start-1:n_end-1, :, 0].flatten() + yhat_sequence)
		NB - if n_input=14, this is 15th iteration in the sequence. This is therefore correct for the next
		5 days prediction after 14 timesteps:
		actuals.append(test[n_start:n_end, :, 0])'''
		n_end = n_start + n_out
		if n_end <= len(test): #Allows for the EoF
			actuals.append(test[n_start:n_end, :, 0])
			#temp_yhat.append(yhat_sequence)
			#temp_nstart.append(test[n_start - 1, :, 0])
			predictions.append(test[n_start - 1:n_end - 1, :, 0].flatten() + yhat_sequence)

		#Tested and working - day before first prediction day. This is the basis to predicting all 5 days:
		# prediction day - 1 + tomorrow's difference yhat = tomorrow's prediction
		# Day 1 undifferenced prediction is now added to day 2's differenced yhat to get day 2
		# And so on until the number of days is achieved - based on day 1 - given that will not have
		# ground truth in production but will have to base 5 day predictions off day 1 changes onward.
		# Finally, this outputs n_out+1. To finalize the predictions, we take predictions_ff[:, 1:]
			predictions_ff.append(test[n_start - 1, :, 0]  + [sum(yhat_sequence[0:x:1]) for x in range(0, n_out+1)])
		n_start += 1

	# evaluate predictions days for each week
	actuals = array(actuals)
	actuals = actuals.reshape(-1, actuals.shape[1])
	predictions = array(predictions)
	predictions_ff = array(predictions_ff)
	predictions_ff = predictions_ff.reshape(-1, predictions_ff.shape[1])
	predictions_ff = predictions_ff[:, 1:] #takes columns 1 - n_out, since the above sequence appends column 0 before the cumulative

	#temp_yhat = array(temp_yhat)
	#temp_nstart = array(temp_nstart)


	#predictions_diff = array(predictions_diff) #can remove
	#act_inv = array(act_inv)
	score, scores = evaluate_forecasts(actuals, predictions)
	return score, scores, actuals, predictions, history


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


	#Test Set###############
	#Only use this code if testing for accuracy
	#import random
	#X = np.array([random.randint(1,500) for x in range(0,1000)])
	#X.sort()
	#dataset = pd.DataFrame(X.reshape(X.shape[0], 1))
	#End Test Set###########

dataset = process_data('AAPL')



#Pulls the close column only and reshapes using a one interval step
dataset_array = np.array(dataset.iloc[:, 0]) #dataset.iloc[:, 0]
df = reshape_dataset(dataset_array, 1)

# evaluate model and get scores
n_input = 15
n_out = 5
score, scores, actuals, predictions, history = evaluate_model(df, n_input, n_out)
# summarize scores
summarize_scores('cnn', score, scores)
# plot scores
days = ['day1', 'day2', 'day3', 'day4', 'day5']
plt.plot(days, scores, marker='o', label='cnn')
plt.show()


#In plotting the actuals v predictions below, it is possible that the model is
#simply learning a persistance - that is, using the most recent value to make
#the prediction.
act = np.array(actuals)
#day0_act = act[:, 0]
#day0_act = act.reshape(act.shape[0]*act.shape[1])

pred = np.array(predictions)
pred_ff = np.array(predictions_ff)
#day0_pred = pred.reshape(pred.shape[0]*pred.shape[1])

for i in range(0, n_out):
	plt.plot (act[-25:, i], color='blue', label='actual')
	plt.plot (pred[-25:, i], color='orange', label='prediction')
	plt.plot(pred_ff[-25:, i], color='red', label='prediction ff')
	plt.title ("Actual v Prediction: Day " + str(i+1))
	plt.legend()
	plt.show ()

#Prints the most recent 10 days predictions
plt.plot (act[-2:, :].flatten(), color='blue', label='actual')
plt.plot (pred[-2:, :].flatten(), color='orange', label='prediction')
plt.plot(pred_ff[-2:, :].flatten(), color='red', label='prediction ff')
plt.legend()
plt.show ()

plt.plot (pd.Series(act[-1:, :].flatten()), color='blue', label='actual')
plt.show()
#Needs a bigger networks for example - more epochs + standardization - relu
#is not working at all.