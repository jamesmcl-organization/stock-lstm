#Review more moving to class then begin hyperp tuning
from math import sqrt
from numpy import split, array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from sklearn.preprocessing import  StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime, date
from importlib import reload
import equity_classes
reload(equity_classes)
from equity_classes import classes as cl

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
def build_model(train, test, n_input, n_out, interval):

	train_scale, scaler = aapl.create_scaler(np.array(aapl.get_difference(train, interval))) #creates the scaler on train
	test_scale = aapl.apply_scaler(np.array(aapl.get_difference(test, interval)), scaler) #applies the scaler to test

	train_x, train_y = aapl.to_supervised(train_scale, n_input, n_out)
	test_x, test_y = aapl.to_supervised(test_scale, n_input, n_out)

	# define parameters
	verbose, epochs, batch_size = 1, 10, 4
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
	model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data=(test_x, test_y))

	plt.plot(model.history.history [ 'loss' ], color='blue')
	plt.plot(model.history.history [ 'val_loss' ], color='orange')
	plt.title("Model Train vs Val Loss: ")
	plt.legend(['train', 'validation'], loc='upper right')

	return model, scaler

	# evaluate a single model
def evaluate_model(df, n_input, n_out, train_pct, interval):

	train, test = aapl.split_dataset(df, train_pct)
	model, scaler = build_model(train, test, n_input, n_out, interval)
	history = [x for x in train]

	# walk-forward validation over each week
	predictions, actuals, predictions_ff = list(), list(), list()
	n_start = n_input
	for i in range(len(test)):
		# predict the week
		yhat_sequence = aapl.forecast(model, history, n_input, scaler, interval)

		# store the predictions
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

	score, scores = evaluate_forecasts(actuals, predictions)
	return score, scores, actuals, predictions, history, predictions_ff


	#Test Set###############
	#Only use this code if testing for accuracy
	#import random
	#X = np.array([random.randint(1,500) for x in range(0,1000)])
	#X.sort()
	#dataset = pd.DataFrame(X.reshape(X.shape[0], 1))
	#End Test Set###########


aapl = cl.prepare_univariate_rnn('AAPL') #instantiate the object
ds = aapl.process_data()
ds_array = np.array(ds.iloc[:, 0])
df = aapl.reshape_dataset(ds_array, 1)


# evaluate model and get scores
n_input = 15
n_out = 5
train_pct = 0.7
interval = 1


score, scores, actuals, predictions, history, predictions_ff = evaluate_model(df, n_input, n_out, train_pct, interval)
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

pred = np.array(predictions)
pred_ff = np.array(predictions_ff)

for i in range(0, n_out):
	plt.plot (act[-25:, i], color='blue', label='actual')
	plt.plot (pred[-25:, i], color='orange', label='prediction')
	plt.plot(pred_ff[-25:, i], color='red', label='prediction ff')
	plt.title ("Actual v Prediction: Day " + str(i+1))
	plt.legend()
	plt.show ()

#Prints the most recent 10 days predictions
plt.plot (act[-1:, :].flatten(), color='blue', label='actual')
plt.plot (pred[-1:, :].flatten(), color='orange', label='prediction')
plt.plot(pred_ff[-1:, :].flatten(), color='red', label='prediction ff')
plt.legend()
plt.show ()

