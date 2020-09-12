#Review more moving to class then begin hyperp tuning
# multi headed multi-step cnn
from math import sqrt
from numpy import split, array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from keras.utils.vis_utils import plot_model
from keras.layers.merge import concatenate
from matplotlib import pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Activation, Input
from keras.layers.convolutional import Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from keras.constraints import maxnorm
from sklearn.preprocessing import  StandardScaler, MinMaxScaler, RobustScaler
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
def build_model(train, test, config):
	#config = cfg_list[0]
	#n_diff=1

	n_input, n_nodes, n_epochs, n_batch, n_diff, n_out, n_lr, n_actfn = config

	train_scale, scaler, scaler_y = aapl.create_scaler(np.array(aapl.get_difference(train, n_diff)))
	#x = np.array(aapl.get_difference(train, n_diff))
	#a, b, c = aapl.create_scaler(x)
	#train_scale, scaler, scaler_y = aapl.create_scaler(x)
	test_scale = aapl.apply_scaler(np.array(aapl.get_difference(test, n_diff)), scaler,scaler_y)  # applies the scaler to test

	train_x, train_y = aapl.to_supervised(train_scale, n_input, n_out)
	test_x, test_y = aapl.to_supervised(test_scale, n_input, n_out)

	# define parameters
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
	decay_rate = n_lr / n_epochs
	ADAM = Adam(lr=n_lr, decay=decay_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
	model.compile(loss='mse', optimizer=ADAM)
	# plot the model
	plot_model(model, show_shapes=True, to_file='multiheaded_cnn.png')
	# fit network
	#input_data = [train_x[:,:,i].reshape((train_x.shape[0],n_timesteps,1)) for i in range(n_features)]
	#model.fit(input_data, train_y, epochs=n_epochs, batch_size=n_batch, verbose=1)

	input_train = [train_x[:, :, i].reshape((train_x.shape[0], n_timesteps, 1)) for i in range(n_features)]
	input_test = [test_x[:, :, i].reshape((test_x.shape[0], n_timesteps, 1)) for i in range(n_features)]
	# fit network
	model.fit(input_train, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0, validation_data=(input_test, test_y))

	return model, scaler, scaler_y



	# evaluate a single model
def evaluate_model(data, n_train, config):

	#config = cfg_list[0]

	n_input, n_nodes, n_epochs, n_batch, n_diff, n_out, n_lr, n_actfn = config
	train, test = aapl.split_dataset(data, n_train)
	model, scaler, scaler_y = build_model(train, test, config)
	history = [x for x in train]

	# walk-forward validation over each week
	predictions, actuals, predictions_ff = list(), list(), list()
	n_start = n_input
	for i in range(len(test)):
		# predict the week
		yhat_sequence = aapl.forecast_mh_cnn(model, history, n_input, scaler, scaler_y, n_diff)

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
			predictions.append(test[n_start - n_diff:n_end - n_diff, :, 0].flatten() + yhat_sequence)

		#Tested and working - day before first prediction day. This is the basis to predicting all 5 days:
		# prediction day - 1 + tomorrow's difference yhat = tomorrow's prediction
		# Day 1 undifferenced prediction is now added to day 2's differenced yhat to get day 2
		# And so on until the number of days is achieved - based on day 1 - given that will not have
		# ground truth in production but will have to base 5 day predictions off day 1 changes onward.
		# Finally, this outputs n_out+1. To finalize the predictions, we take predictions_ff[:, 1:]
			predictions_ff.append(test[n_start - n_diff, :, 0]  + [sum(yhat_sequence[0:x:1]) for x in range(0, n_out+1)])
		n_start += 1

	# evaluate predictions days for each week
	actuals = array(actuals)
	actuals = actuals.reshape(-1, actuals.shape[1])
	predictions = array(predictions)
	predictions_ff = array(predictions_ff)
	predictions_ff = predictions_ff.reshape(-1, predictions_ff.shape[1])
	predictions_ff = predictions_ff[:, 1:] #takes columns 1 - n_out, since the above sequence appends column 0 before the cumulative

	score, scores = evaluate_forecasts(actuals, predictions)
	#return score, scores, actuals, predictions, history, predictions_ff
	return scores


# score a model, return None on failure
def repeat_evaluate(data, config, n_train, n_repeats=3):
	# convert config to a key
	key = str(config)
	# fit and evaluate the model n times
	#scores = [walk_forward_validation(data, n_test, config) for _ in range(n_repeats)]
	scores = [evaluate_model(data, n_train, config)  for _ in range(n_repeats)]
	# summarize score from the repeats of each config
	result = np.mean(scores)
	print('> Model[%s] %.3f' % (key, result))
	return (key, result)

# grid search configs
def grid_search(data, cfg_list, n_train):
	# evaluate configs
	scores = [repeat_evaluate(data, cfg, n_train) for cfg in cfg_list]
	# sort configs by error, asc
	scores.sort(key=lambda tup: tup[1])
	return scores

aapl = cl.parent_rnn('AAPL') #instantiate the object
ds = aapl.process_data()
ds = ds.drop(['day', 'ticker'], axis=1)
#'volume_delta', 'prev_close_ch', 'prev_volume_ch',
#			  'macds', 'macd', 'dma', 'macdh', 'ma200'], axis=1)

ds_array = np.array(ds.iloc[:, :]) #included for the univariate solutions
data = aapl.reshape_dataset(ds_array, 1)

n_train = 0.8


cfg_list = aapl.model_configs()

scores = grid_search(data, cfg_list, n_train)
print('done')

#list top configs
for cfg, error in scores[:5]:
	print(cfg, error)
