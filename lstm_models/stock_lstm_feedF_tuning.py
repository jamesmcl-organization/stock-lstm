import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
# import pandas_datareader as pdr
# from numpy.core._multiarray_umath import ndarray
from matplotlib import pyplot as plt

sns.set (style="darkgrid", color_codes=True)
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,  LSTM, Dropout
from sklearn.preprocessing import  StandardScaler
from tensorflow.keras.optimizers import Adam
import matplotlib.dates as mdates
from tensorflow.keras.regularizers import L1L2
import os


def difference_pct(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = (dataset[i] - dataset[i - interval]) / dataset[i - interval]
        diff.append(value)
    return pd.Series(diff)


def bool_change_pct(y):
    df = pd.DataFrame.from_records (np.row_stack (y))
    return df.applymap(lambda x: 1 if x > 0 else 0)

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
        #in_start += predsteps this will cause a no overlap prediction
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

    #X_scaler = MinMaxScaler (feature_range=(0, 1))
    X_scaler = StandardScaler ()
    X_scaler = X_scaler.fit (X_t)
    #y_scaler = MinMaxScaler (feature_range=(0, 1))
    y_scaler = StandardScaler ()
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


def inverse_difference_pct(history, yhat, interval=1):
    return (yhat * history[-interval]) + history[-interval]
    #return history[-interval]


# inverse scaling for a forecasted value
def invert_scale(scaler, yhat):
    inverted = scaler.inverse_transform (yhat)
    return inverted[0, :]


def fit_lstm(X_train, y_train, X_dev, y_dev, batch_size, nb_epoch, neurons, reg, opt):

    #tboard_params = get_tensorboard ()
    # verbose, epochs, batch_size = 1, 100, 64

    n_timesteps, n_features, n_outputs = X_train.shape [ 1 ], X_train.shape [ 2 ], y_train.shape [ 1 ]

    model = Sequential ()
    model.add (LSTM (units=neurons, activation='tanh', batch_input_shape=(batch_size, n_timesteps, n_features),return_sequences=True, stateful=True, bias_regularizer=reg))
    model.add (Dropout (0.4))
    #model.add (Dense (100, activation='tanh'))
    model.add (LSTM (units=neurons//2, activation='tanh', return_sequences=True))
    model.add (LSTM (units=neurons // 3, activation='tanh'))
    model.add (Dense (5))

    ADAM = Adam (opt, beta_1=0.9, beta_2=0.999, amsgrad=False)
    #model.compile (loss='mean_squared_error', optimizer='adam')
    model.compile (loss='mean_squared_error', optimizer=ADAM)
    #model.summary()

    train_loss, val_loss = [], []
    #train_loss, val_loss = np.array ([ ]), np.array ([ ])

    for i in range (nb_epoch):
        #model.fit (X_train, y_train, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
        model.fit (X_train, y_train, epochs=1, batch_size=batch_size, verbose=1, shuffle=False, validation_data=(X_dev, y_dev))

        train_loss.append(model.history.history [ 'loss' ])
        val_loss.append (model.history.history [ 'val_loss' ])

        model.reset_states ()

    #Steps
    #Return loss as a numpy array
    #capture the updates and append them to the original loss list
    loss = np.hstack([np.array(train_loss), np.array (val_loss)])

    return model, loss


def plot_loss(loss, caption):
    #loss = np.array (loss)
    #loss_arr = loss.reshape (loss.shape [ 0 ], loss.shape [ 1 ])
    plt.plot (loss [ :, 0 ], color='blue')
    plt.plot (loss [ :, 1 ], color='orange')
    plt.title ("Model Train vs Val Loss: " + str(caption))
    plt.ylabel ('Loss')
    plt.xlabel ('Epoch')
    plt.legend ([ 'train', 'validation' ], loc='upper right')
    #plt.show ()



# make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, X.shape[0], X.shape[1])
    yhat = model.predict(X, batch_size=batch_size)
    return yhat

def chart_results(predsteps, predictions, y_dev, plot_days):
    days = [ "Day" + str (i) for i in range (1, predsteps + 1) ]
    plot_days = [ "Day" + str (i) for i in range (1, plot_days + 1) ]
    df_pred = pd.DataFrame.from_records(np.row_stack(predictions), columns=[ i for i in days ]).stack ().reset_index ()
    df_actual = pd.DataFrame.from_records (np.row_stack(y_dev), columns=[ i for i in days ]).stack ().reset_index ()
    df_pred.rename (columns={'level_0': 'iteration', 'level_1': 'day', 0: 'pred'}, inplace=True)

    df_pred [ 'actual' ] = df_actual [ 0 ]
    df_pred = df_pred [ df_pred [ 'day' ].isin (plot_days) ]
    df_pred.reset_index(inplace=True)

    fig, axes = plt.subplots (figsize=(16, 8))

    axes.plot (df_pred[ 'pred' ], label='predicted', linewidth=1)
    axes.plot (df_pred[ 'actual' ], label='actual', linewidth=1)

    axes.legend ([ 'Pred', 'Actual'], loc='upper right')
    plt.show ()

    return df_pred


# execute the experiment
def process_data():
    project_path = r'/home/ubuntu/stock_nn'
    os.chdir (project_path)

    # Import the Dataset
    df = pd.read_csv ('aapl.csv', low_memory=False)
    df [ 'date' ] = pd.to_datetime (df.date)
    df.index = df [ 'date' ]
    df = df.iloc[-1000:]

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
    'volume_delta', 'prev_close_ch', 'prev_volume_ch',
                          'macds', 'macd', 'dma', 'macdh', 'ma200'], axis=1)
    df_close = df_close.sort_index (ascending=True, axis=0)

#Move the target variable to the end of the dataset so that it can be split into X and Y for Train and Test
    cols = list(df_close.columns.values) #Make a list of all of the columns in the df
    cols.pop(cols.index('close')) #Remove outcome from list
    df_close = df_close[['close']+cols] #Create new dataframe with columns in correct order

    df_close = df_close.dropna ()

    return df_close


def update_model(model, X_train, y_train, batch_size, updates):

    train_loss, val_loss = [],[]
    for i in range(updates):
        model.fit (X_train, y_train, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
    return model

# run experiment
# run a repeated experiment
def experiment(repeats, diff_values, df_close, timesteps, predsteps, batch_size, nb_epoch, neurons, reg, opt):
    X, y = timeseries_to_supervised (diff_values.values, timesteps, predsteps)
    X_raw, y_raw = timeseries_to_supervised (df_close.values, timesteps, predsteps)
    # _dates = pd.DataFrame (df_close.reset_index (level=[ 'date' ]) [ 'date' ])
    # _, _dates = timeseries_to_supervised (_dates.values, timesteps, predsteps)

    X_train, y_train, X_dev, y_dev = split_data (X, y)
    _, y_train_raw, _, y_dev_raw = split_data (X_raw, y_raw)

    X_train_norm, y_train_norm, X_dev_norm, y_dev_norm, X_scaler, y_scaler = scale_data (X_train, y_train, X_dev, y_dev)
    error_scores = list ()
    for r in range (repeats):
        lstm_model, model_loss = fit_lstm (X_train_norm, y_train_norm, X_dev_norm, y_dev_norm, batch_size, nb_epoch, neurons, reg, opt)
        plot_loss (model_loss, "Initial Training")
        #if r == 0:
        #    print (lstm_model.summary ())
        print ('%d) Train RMSE= %f, Dev RMSE= %f' % (r, model_loss [ -1:, 0 ], model_loss [ -1:, 1 ]))
        plt.show () #Display the initial loss

        X_train_norm_copy, y_train_norm_copy = np.copy (X_train_norm), np.copy (y_train_norm)
        X_dev_norm_copy, y_dev_norm_copy = np.copy (X_dev_norm), np.copy (y_dev_norm)

        # forecast test dataset
        predictions, yhat_inver, pred_yinv, pred_yinver, y_invscale = [ ], [ ], [ ], [ ], [ ]
        updated_loss = [ ]
        for i in range (len (X_dev_norm)):
            if i > 0: #Model updates once for each sample in train
                updated_model = update_model (lstm_model, X_train_norm_copy, y_train_norm_copy, 1, updates)

            X_, y_ = X_dev_norm [ i, :, : ].astype (np.float32), y_dev_norm [ i, : ].astype (np.float32)
            yhat = forecast_lstm (lstm_model, 1, X_)
            print('Update yhat loss: ' + str(mean_squared_error(yhat, y_dev_norm [ [ i ] ].reshape (y_dev_norm [ [ i ] ].shape [ 0 ],
                                                                                                y_dev_norm [ [ i ] ].shape [ 1 ]))))

            yhat = invert_scale (y_scaler, yhat.reshape (yhat.shape [ 0 ], yhat.shape [ 1 ]))
            yhat = yhat.reshape (1, yhat.shape [ 0 ])
            yhat_inver.append (yhat)  # Use this for the Classification Report
            predictions.append (inverse_difference_pct (y_raw, yhat, len (y_dev_norm) + 1 - i))

            y_inv = invert_scale (y_scaler, y_dev_norm [ [ i ] ].reshape (y_dev_norm [ [ i ] ].shape [ 0 ],
                                                                          y_dev_norm [ [ i ] ].shape [ 1 ]))
            y_inv = y_inv.reshape (1, y_inv.shape [ 0 ])
            pred_yinver.append (y_inv)  # Use this for the Classification Report
            pred_yinv.append (inverse_difference_pct (y_raw, y_inv, len (y_dev_norm) + 1 - i))

            X_train_norm_copy = np.concatenate ((X_train_norm_copy, X_dev_norm [ i, :, : ].reshape (1, timesteps,
                                                                                                    -1)))  # Reshape to 1, timesteps, -1=num features
            y_train_norm_copy = np.concatenate (
                (y_train_norm_copy, y_dev_norm [ i, : ].reshape (1, -1)))  # Reshape to 1, -1=num predsteps

        rmse = np.sqrt (mean_squared_error (np.asarray (pred_yinv).flatten (), np.asarray (predictions).flatten ()))
        error_scores.append (rmse)
        df_pred = chart_results (predsteps, predictions, pred_yinv, 1)

        return error_scores

#   loss_arr = np.hstack ([ np.array (loss[:,0]), np.array (loss[:,1])])
#   plot_loss (loss, "Walk Forward Training")
#plt.show()



repeats = 1
#timesteps = [5, 10, 15, 25, 50]
#regularizers = [L1L2(l1=0.0, l2=0.0), L1L2(l1=0.02, l2=0.0), L1L2(l1=0.01, l2=0.0), L1L2(l1=0.0, l2=0.01), L1L2(l1=0.01, l2=0.01)]
#opt = [0.0001, 0.0008, 0.002, 0.005, 0.01, 0.05]

timesteps = [25]
regularizers = [L1L2(l1=0.01, l2=0.01)]
opt = [0.0009]

predsteps = 5
train_pct = 0.90
updates = 2
batch_size = 1
nb_epoch = 20
neurons = 100

results = pd.DataFrame()

df_close = process_data()
diff_values = df_close.apply(difference_pct, args=[1])

for reg in regularizers:
    for ts in timesteps:
        for op in opt:
            name = ('l1 %.2f,l2 %.2f' % (reg.l1, reg.l2) + " TS: " + str(ts) + " OPT: " + str(op))
            print(str(name))
            results[str(name)] = experiment(repeats, diff_values, df_close, ts, predsteps, batch_size, nb_epoch, neurons, reg, op)

#print(results.describe())
	# save boxplot
#results.boxplot()
plt.savefig('experiment_reg_input.png')
plt.show()

results.to_csv (r'/home/ubuntu/stock_nn/tuning_results.csv', index=True, header=True)

results_t = results.T
results_t.to_csv (r'/home/ubuntu/stock_nn/tuning_results_t.csv', index=True, header=True)

project_path = r'/home/ubuntu/stock_nn'
os.chdir (project_path)

    # Import the Dataset
df = pd.read_csv ('tuning_results_t.csv', low_memory=False)

# summarize results
