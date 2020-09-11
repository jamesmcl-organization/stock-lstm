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

from keras.utils import np_utils
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.metrics import classification_report, multilabel_confusion_matrix, balanced_accuracy_score

from importlib import reload
import equity_classes
reload(equity_classes)
from equity_classes import classes as cl
#CUDA_VISIBLE_DEVICES=0, 1


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

dummy_y, encoded_y, encoder = aapl_reg.encode_y(nday_chg_label)

aapl_reg.get_exp_charts(nday_chg_label)
aapl_reg.get_exp_charts(intraday_max_label)

X_train, X_test, y_train, y_test = train_test_split(X_classical, encoded_y, test_size=0.3, random_state=101)

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


rf_classifier = RandomForestClassifier(criterion = 'entropy', random_state = 42)

rf_pipeline = Pipeline([
    ('scaler',RobustScaler()),
	('pca', PCA(n_components=0.97)),
    ('kc', rf_classifier)
])

rf_validator = RandomizedSearchCV(estimator = rf_classifier,
								  cv=3,
								  param_distributions=random_grid,
								  n_iter = 50,
								  verbose=1,
								  random_state=42,
								  n_jobs=-1)

sample_weights = compute_sample_weight('balanced', y_train)
rf_validator.fit(X_train, y_train, sample_weight=sample_weights)

aapl_reg.get_results(rf_validator, X_test, y_test, encoder)






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

dummy_y, encoded_y, encoder = aapl_reg.encode_y(nday_chg_label)



X_train, X_test, y_train, y_test = train_test_split(X_classical, encoded_y, test_size=0.3, random_state=101)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


# A parameter grid for XGBoost
random_grid = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5],
        'loss':['hinge','log','modifier_huber','squared_hinge','perceptron'],
        'penalty':['li','l2','elasticnet'],
        'alpha':[0.0001, 0.001,0.01,0.1,1,10,100,1000],
        'learnin_rate':['constant','optimal','invscaling','adaptive'],
        'class_weight':[{0.3,0.5,0.2},{0.3,0.4,0.3}],
        'eta0':[1,10,100]
        }

xgb_classifier = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='multi:softmax',
                    silent=True, nthread=1)


xgb_pipeline = Pipeline([
    ('scaler',RobustScaler()),
	('pca', PCA(n_components=0.97)),
    ('kc', xgb_classifier)
])


xgb_validator = RandomizedSearchCV(xgb_classifier,
                                   param_distributions=random_grid,
                                   n_iter=50,
                                   #scoring='roc_auc',
                                   n_jobs=-1,
                                   cv=3,
                                   verbose=1,
                                   random_state=42 )

# Here we go
start_time = timer(None) # timing starts from this point for "start_time" variable

sample_weights = compute_sample_weight('balanced', y_train)
xgb_validator.fit(X_train, y_train, sample_weight=sample_weights)

timer(start_time) # timing ends here for "start_time" variable
aapl_reg.get_results(xgb_validator, X_test, y_test, encoder)









print('\n All results:')
print(random_search.cv_results_)
print('\n Best estimator:')
print(random_search.best_estimator_)
print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
print(random_search.best_score_ * 2 - 1)
print('\n Best hyperparameters:')
print(random_search.best_params_)
results = pd.DataFrame(random_search.cv_results_)
results.to_csv('xgb-random-grid-search-results-01.csv', index=False)

y_test = random_search.predict_proba(test)
results_df = pd.DataFrame(data={'id':test_df['id'], 'target':y_test[:,1]})
results_df.to_csv('submission-random-grid-search-xgb-porto-01.csv', index=False)