import pandas as pd
import numpy as np
import matplotlib as plt
from matplotlib import pyplot as plt
from numpy import split, array
from pandas import read_csv
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


headers = pd.read_csv (r'/home/ubuntu/stock_lstm/export_files/headers.csv')
df = pd.read_csv (r'/home/ubuntu/stock_lstm/export_files/stock_history.csv', header=None, names=list(headers))

#df.replace('?', nan, inplace=True)

# Extract the close for 'AAPL' only
    history = df[df['ticker'] == 'AAPL'].sort_index(ascending=True)
    history.index.name = 'date'

history['macdh'].plot()
plt.show()

for i in headers:
	history[i].plot(title=str(i))
	plt.show()

for i in headers:
	history[i].hist(bins=30)
	plt.title(str(i))
	plt.show()

plt.figure(figsize=(20, 20))
#axes [ 2 ].plot (aapl.index, aapl [ 'prev_close_ch' ], color='red')
for i in range(len(history.columns)):
	# create subplot
	plt.subplot(len(history.columns), 1, i+1)
	# get variable name
	name = history.columns[i]
	# plot data
	plt.plot(history[name])
	# set title
	plt.title(name, y=0)
	# turn off ticks to remove clutter
	plt.yticks([])
	plt.xticks([])
plt.show()

#Chart out the ACF and PACF charts to determine the correlations between the current
#timestep and previous timesteps - in order to select an appropriate time step value.


#Overall lag autocorrelation
close_autocorr = history['close'].iloc[-365:,].autocorr(lag=1)
print(close_autocorr)


def plot_acf_pacf(data, lags):
	plt.figure()
	# acf
	axis = plt.subplot(2, 1, 1)
	plot_acf(data, ax=axis, lags=lags)
	# pacf
	axis = plt.subplot(2, 1, 2)
	plot_pacf(data, ax=axis, lags=lags)
	# show plot
	plt.show()

#Select the last year of data to investigate autocorrelation
autocorr = array(history['close'].iloc[-365:,]).flatten() #convert close to series
plot_acf_pacf(autocorr, 20)
#No evidence of autocorrelation

#Select all close data to look for seasonality
seasonality = array(history['close']).flatten() #convert close to series
plot_acf_pacf(seasonality, 800)
#No real evidence of seasonality over the past 800 days
# - The time series shows a strong temporal dependence (autocorrelation)
# that decays linearly or in a similar pattern - could be evidence of a random
# walk.

#Now consider the trend - to look at trend, remove seasonality through
#differencing
def difference_pct(dataset, interval):
    diff = list()
    for i in range(interval, len(dataset)):
        value = (dataset[i] - dataset[i - interval]) ##/ dataset[i - interval]
        diff.append(value)
    return pd.Series(diff)

trend = difference_pct(history['close'], interval=800)
plot_acf_pacf(trend, 800)
#Interestingly, when differencing is applied, there does appear to be
#trend that reveals itself.