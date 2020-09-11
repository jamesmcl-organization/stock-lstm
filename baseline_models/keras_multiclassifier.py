from matplotlib import pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime, date
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from keras.utils import np_utils
from keras.regularizers import l1, l2, l1_l2
from keras.initializers import glorot_uniform, zeros, glorot_normal
from keras import initializers
from keras.constraints import maxnorm
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.metrics import classification_report, multilabel_confusion_matrix, balanced_accuracy_score

from importlib import reload
import equity_classes
reload(equity_classes)
from equity_classes import classes as cl
CUDA_VISIBLE_DEVICES=0, 1


'''{'n_nodes': 100, 'n_lr': 0.001, 'init': 'uniform', 'epochs': 200, 'dropout': 0, 'batch_size': 32, 'activation': 'tanh'}
17/17 [==============================] - 0s 742us/step - loss: 2.2520 - accuracy: 0.2900
loss :  2.2519631385803223
accuracy :  0.2899628281593323'''




aapl_reg = cl.prepare_classical('AAPL') #instantiate the object
dataset = aapl_reg.process_data()
dataset = dataset.drop(['adj close', 'day', 'ticker'], axis=1)
df_reshape = aapl_reg.reshape_dataset(np.array(dataset), 1)

X, y = aapl_reg.to_supervised_classical(df_reshape, 15, 5)


X_classical = pd.DataFrame(aapl_reg.reshape_X_classical(X)) #Reshapes X into 1 row and all columns for the features
y_classical = aapl_reg.reshape_y_classical(y, n_out=5) #Reshapes y to calculate % change


nday_chg, intraday_max = aapl_reg.get_chg_pc(y_classical)

nday_chg_label = pd.DataFrame.from_records(aapl_reg.get_chg_pc_label(nday_chg), columns = ['nday_chg', 'nday_chg_label'])
intraday_max_label = pd.DataFrame.from_records(aapl_reg.get_chg_pc_label(intraday_max), columns = ['intraday_max', 'intraday_max_label'])


aapl_reg.get_exp_charts(nday_chg_label)
aapl_reg.get_exp_charts(intraday_max_label)



dummy_y, encoded_y, encoder = aapl_reg.encode_y(nday_chg_label)

X_train, X_test, y_train, y_test = train_test_split(X_classical, encoded_y, test_size=0.3, random_state=101)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Function to create model, required for KerasClassifier
def create_model(n_nodes=100, n_lr=0.001, batch_size=16,
		init='uniform', epochs=50, dropout=0.5, activation='tanh',
		n_input=X_train.shape[1], n_out=5):

#n_out=y_train.shape[1],
	model = Sequential()
	model.add(Dense(units=n_nodes, input_dim=n_input, kernel_initializer=init, activation=activation))
	model.add(Dropout(dropout))
	model.add(Dense(units=n_nodes // 2, kernel_initializer=init, activation=activation))
	model.add(Dropout(dropout))
	model.add(Dense(n_out, activation='softmax'))
	# Compile model
	decay_rate = n_lr / epochs
	# opt = optimizer_val(lr=n_lr, decay=decay_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
	ADAM = Adam(lr=n_lr, decay=decay_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
	model.compile(optimizer=ADAM, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	#model.fit(X, y, epochs=n_epochs, batch_size=n_batch, verbose=1)
	return model


param_grid = \
	{
'n_nodes': [50, 100, 200, 300],
'n_lr': [0.005, 0.01, 0.05, 0.10, 0.001],
'batch_size':[16, 32, 64, 128, 256],
'init':['uniform', 'normal', 'zeros'],
'activation': ['tanh', 'relu'],
'epochs':[100, 200, 300],
'dropout': [0.5, 0.2, 0.1, 0]
}

my_classifier = KerasClassifier(build_fn=create_model, verbose=0)



pipeline = Pipeline([
    ('scaler',RobustScaler()),
	('pca', PCA(n_components=0.97)),
    ('kc', KerasClassifier(build_fn=create_model, verbose=1))
])




# The scorers can be either be one of the predefined metric strings or a scorer
# callable, like the one returned by make_scorer
from sklearn.metrics import make_scorer, f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import accuracy_score

scorer = {'accuracy': make_scorer(accuracy_score),
           #'precision': make_scorer(precision_score, average = 'macro'),
           'recall': make_scorer(recall_score, average = 'macro'),
           'f1_macro': make_scorer(f1_score, average = 'macro'),
           'f1_weighted': make_scorer(f1_score, average = 'weighted')}


validator = RandomizedSearchCV(my_classifier,
						 cv=3,
                         param_distributions=param_grid,
						n_iter = 35,
						 n_jobs=-1,
						scoring=scorer,
						refit='f1_weighted',
						return_train_score=True)



sample_weights = compute_sample_weight('balanced', y_train)
validator.fit(X_train, y_train, sample_weight=sample_weights)


aapl_reg.get_results(validator, X_test, y_test, encoder)