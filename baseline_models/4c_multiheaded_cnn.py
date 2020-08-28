# multi headed multi-step cnn
import pandas as pd
from math import sqrt
from numpy import split, array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.utils.vis_utils import plot_model
from keras.layers.merge import concatenate
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

def split_dataset(data, train_pct):

	'''performs the splitting of the already reshaped dataset'''

	train_weeks = int(data.shape[0] * train_pct)
	train, test = data[0:train_weeks, :, :], data[train_weeks:, :, :]

	return train, test

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


# plot training history
def plot_history(history):
	# plot loss
	plt.subplot(2, 1, 1)
	plt.plot(history.history['loss'], label='train')
	plt.plot(history.history['val_loss'], label='test')
	plt.title('loss', y=0, loc='center')
	plt.legend()
	# plot rmse
	plt.subplot(2, 1, 2)
	plt.plot(history.history['rmse'], label='train')
	plt.plot(history.history['val_rmse'], label='test')
	plt.title('rmse', y=0, loc='center')
	plt.legend()
	plt.show()


# train the model
def build_model(train, n_input, n_out):
	# prepare data
	train_x, train_y = to_supervised(train, n_input, n_out)
	# define parameters
	verbose, epochs, batch_size = 1, 500, 32
	n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
	# create a channel for each variable
	in_layers, out_layers = list(), list()
	for _ in range(n_features): #Runs a model through for each one of the features
		inputs = Input(shape=(n_timesteps,1))
		conv1 = Conv1D(32, 3, activation='tanh')(inputs)
		conv2 = Conv1D(32, 3, activation='tanh')(conv1)
		pool1 = MaxPooling1D()(conv2)
		flat = Flatten()(pool1)
		# store layers
		in_layers.append(inputs) #appends each
		out_layers.append(flat) #learned features from each sequence
	# merge heads
	merged = concatenate(out_layers)
	# interpretation
	dense1 = Dense(200, activation='tanh')(merged)
	dense2 = Dense(100, activation='tanh')(dense1)
	outputs = Dense(n_outputs)(dense2)
	model = Model(inputs=in_layers, outputs=outputs)
	# compile model
	model.compile(loss='mse', optimizer='adam')
	# plot the model
	plot_model(model, show_shapes=True, to_file='multiheaded_cnn.png')
	# fit network
	input_data = [train_x[:,:,i].reshape((train_x.shape[0],n_timesteps,1)) for i in range(n_features)]
	model.fit(input_data, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
	return model

# make a forecast
def forecast(model, history, n_input):
	# flatten data
	data = array(history)
	data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
	# retrieve last observations for input data
	input_x = data[-n_input:, :]
	# reshape into n input arrays
	input_x = [input_x[:,i].reshape((1,input_x.shape[0],1)) for i in range(input_x.shape[1])]
	# forecast the next week
	yhat = model.predict(input_x, verbose=0)
	# we only want the vector forecast
	yhat = yhat[0]
	return yhat


# evaluate a single model
def evaluate_model(train, test, n_input, n_out):
	# fit model
	model = build_model(train, n_input, n_out)
	# history is a list of weekly data
	history = [x for x in train]
	# walk-forward validation over each week
	predictions, actuals = list(), list()
	for i in range(len(test)):
		# predict the week
		yhat_sequence = forecast(model, history, n_input)
		# store the predictions
		predictions.append(yhat_sequence)
		# get real observation and add to history for predicting the next week
		history.append(test[i, :])
		# This next step can come out once the model is confirmed. It pulls the actuals
		########################################################################
		n_start = n_input
		n_end = n_start + n_out
		if n_end <= len(test):  # Allows for the EoF
			actuals.append(test[n_start:n_end, :, 0])
		n_start += 1
	#########################################################################
	# evaluate predictions days for each week
	actuals = array(actuals)
	actuals = actuals.reshape(-1, actuals.shape[1])
	predictions = array(predictions)
	score, scores = evaluate_forecasts(actuals, predictions)
	return score, scores, actuals, predictions, history



def process_data(ticker):

	headers = pd.read_csv (r'/home/ubuntu/stock_lstm/export_files/headers.csv')
	df = pd.read_csv (r'/home/ubuntu/stock_lstm/export_files/stock_history.csv', header=None, names=list(headers))
	df.index.name = 'date'

	df.reset_index (inplace=True)		#temporarily reset the index to get the week day for OHE
	df['date']= pd.to_datetime(df['date'])
	df.drop_duplicates(['date', 'ticker', 'close'], inplace=True)
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

dataset_array = np.array(dataset)
df = reshape_dataset(dataset_array, 1)
df_diff = np.array(difference(df, interval=1))

train, test = split_dataset(df, 0.8)
train_diff, test_diff = split_dataset(df_diff, 0.8)

#X, y = timeseries_to_supervised(dataset.values, 5, 5)
#train, test = split_data(X, y, 0.9)
# evaluate model and get scores
n_input = 14
n_out = 5
score, scores, actuals, predictions, history = evaluate_model(train, test, n_input, n_out)
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
day0_act = act.reshape(act.shape[0]*act.shape[1])

pred = np.array(predictions)
day0_pred = pred.reshape(pred.shape[0]*pred.shape[1])

for i in range(0, n_input):
	plt.plot (act[:, i], color='blue')
	plt.plot (pred[:, i], color='orange')
	plt.title ("Actual v Prediction: Day " + str(i))
	plt.show ()

#Needs a bigger networks for example - more epochs + standardization - relu
#is not working at all.