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


def get_exp_charts(data):
	'''plots 3 charts:
	Plot 1: histogram of the % change distribution across all categories
	Plot 2: Bar chart counting each category
	Plot 3: histogram grouped by each category'''
	sns.distplot(data.iloc[:, 0], kde=True, bins=100).set_title(data.columns[0])
	plt.show()

	sns.countplot(x=data.iloc[:, 1], data=data).set_title(data.columns[0])
	plt.show()

	fig = plt.figure(figsize=(8, 4))

	labels = data.iloc[:, 1].unique()
	for i in labels:
		sns.distplot(data[data.iloc[:, 1] == i].iloc[:, 0], kde=False, bins=25)
	fig.legend(labels=labels)
	plt.show()


def encode_y(data):
	y = data.iloc[:, 1]
	encoder = LabelEncoder()
	encoder.fit(y)
	encoded_y = encoder.transform(y) #would be sufficient for a binary classifier
	# convert integers to dummy variables (i.e. one hot encoded) - because of multi-classification
	dummy_y = np_utils.to_categorical(encoded_y)
	return dummy_y, encoded_y, encoder


def decode_y(y, encoder):
	'''Takes the dummy_y and returns the original label'''
	#inv_encode = encoder.inverse_transform(np.argmax(y, axis=-1))
	inv_encode = encoder.inverse_transform(y)
	return pd.DataFrame({'y_label': inv_encode})

def decode_yhat(yhat, encoder):
	'''Takes the y and returns the original label and the max probability for the classifier
	Can be applied to yhat output or the dummy_y - columns will require renaming'''
	inv_encode = encoder.inverse_transform(np.argmax(yhat, axis=-1))
	inv_proba = np.max(yhat, axis=-1)
	return pd.DataFrame({'yhat_proba': inv_proba, 'yhat_label': inv_encode})



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


get_exp_charts(nday_chg_label)
get_exp_charts(intraday_max_label)



dummy_y, encoded_y, encoder = encode_y(nday_chg_label)

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


################Model Summary and Parameters####################################
print('The parameters of the best model are: ')
print(validator.best_params_)
results = validator.cv_results_

best_model = validator.best_estimator_.model
metric_names = best_model.metrics_names
metric_values = best_model.evaluate(X_test, y_test)
for metric, value in zip(metric_names, metric_values):
    print(metric, ': ', value)

'''{'n_nodes': 100, 'n_lr': 0.001, 'init': 'uniform', 'epochs': 200, 'dropout': 0, 'batch_size': 32, 'activation': 'tanh'}
17/17 [==============================] - 0s 742us/step - loss: 2.2520 - accuracy: 0.2900
loss :  2.2519631385803223
accuracy :  0.2899628281593323'''


# the result is also a binary label matrix
yhat = best_model.predict(X_test)
yhat_inv = decode_yhat(yhat, encoder)
#np.argmax(model.predict(x), axis=-1)

y_inv = decode_y(y_test, encoder)
df_inv = pd.concat([yhat_inv, y_inv], axis=1)

multilabel_confusion_matrix(df_inv['y_label'], df_inv['yhat_label'])

df_inv['rank'] = df_inv.groupby(['y_label'])['yhat_proba'].transform(
                     lambda x: pd.qcut(x, 5, labels=range(1,6)))

df_inv['rank'] = df_inv['rank'].astype('int32')

print('Multilabel Confusion Matrix (Overall): ' + str(multilabel_confusion_matrix(df_inv['y_label'], df_inv['yhat_label'])))
print('Balanced Accuracy Score (Overall): ' + str(balanced_accuracy_score(df_inv['y_label'], df_inv['yhat_label'])))

rcount = list(df_inv['rank'].unique())
for i in range(1,len(rcount)+1):
	df = df_inv[df_inv['rank'] == i]
	print('Balanced Accuracy Score Rank' + str(i) + ' ' + str(balanced_accuracy_score(df['y_label'], df['yhat_label'])))






X_train, X_test, y_train, y_test = train_test_split(X_classical, y_label_df['change_label'], test_size=0.3, random_state=101)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)



# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


rf_classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 42)

rf_pipeline = Pipeline([
    ('scaler',RobustScaler()),
	('pca', PCA(n_components=0.97)),
    ('kc', rf_classifier)
])



rf_validator = RandomizedSearchCV(estimator = rf_classifier,
						 cv=3,
                         param_distributions=random_grid,
						n_iter = 50,
								  verbose=2,
								  random_state=42,
						 n_jobs=-1)



sample_weights = compute_sample_weight('balanced', y_train)
# Fit the random search model
rf_validator.fit(X_train, y_train)


print('The parameters of the best model are: ')
print(rf_validator.best_params_)


def evaluate(model, X_test, y_test):
	predictions = model.predict(X_test)
	errors = abs(predictions - y_test)
	mape = 100 * np.mean(errors / y_test)
	accuracy = 100 - mape
	print('Model Performance')
	print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
	print('Accuracy = {:0.2f}%.'.format(accuracy))

	return accuracy


base_model = RandomForestClassifier(n_estimators=10, random_state=42)
base_model.fit(X_train, y_train)
base_accuracy = evaluate(base_model, X_test, y_test)


best_random = rf_validator.best_estimator_
random_accuracy = evaluate(best_random, X_test, y_test)



print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))







# {'n_nodes': 200, 'n_lr': 0.005, 'init': 'normal', 'epochs': 200,
# 'dropout': 0.1, 'batch_size': 64, 'activation': 'tanh'}
# validator.best_estimator_ returns sklearn-wrapped version of best model.
# validator.best_estimator_.model returns the (unwrapped) keras model
best_model = validator.best_estimator_.model
metric_names = best_model.metrics_names
metric_values = best_model.evaluate(X_train, y_train)
for metric, value in zip(metric_names, metric_values):
    print(metric, ': ', value)




# Predicting the Test set results
y_pred = classifier.predict(X_test)
#Reverse factorize (converting y_pred from 0s,1s and 2s to Iris-setosa, Iris-versicolor and Iris-virginica
reversefactor = dict(zip(range(6),definitions))
y_test = np.vectorize(reversefactor.get)(y_test)
y_pred = np.vectorize(reversefactor.get)(y_pred)
# Making the Confusion Matrix
print(pd.crosstab(y_test, y_pred, rownames=['Actual Species'], colnames=['Predicted Species']))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
'''