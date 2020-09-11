from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime, date
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, Normalizer
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

n_components = [330, 100] #this is for PCA() - best course of action appears to be normalization of the data

#np.unique(encoded_y)
class_weights = compute_class_weight('balanced', np.unique(y_train), y_train)
weights_dict = dict(enumerate(class_weights))

#from imblearn.ensemble import BalancedRandomForestClassifier

rf_classifier = RandomForestClassifier(criterion = 'entropy', random_state = 42, class_weight=weights_dict)
#rf_classifier = RandomForestClassifier(criterion = 'entropy', random_state = 42)

'''
The PCA() was showing ultimately very poor results. It should be properly tested with a 
proper set of parameters. Using Normalizer might show the most promise.

X_normalized = Normalizer(X, norm='l2')
scaler = Normalizer()
scaler = scaler.fit(X_train)
train_scale = scaler.fit_transform(X_train)

pca = PCA()
pca.fit(train_scale)

plt.figure(1, figsize=(16, 10))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_')
plt.show()

pca = PCA(2)
low_d = pca.fit_transform(train_scale)
plt.scatter(low_d[:,0], low_d[:,1])
plt.show()
'''

rf_pipeline = Pipeline(steps=
[
    #('scaler',RobustScaler()),
    ('scaler',MinMaxScaler()),
	#('pca', PCA(n_components=25)),
    ('kc', rf_classifier)
])

random_grid = {
                #'pca__n_components': n_components,
                'kc__n_estimators': n_estimators,
                'kc__max_features': max_features,
                'kc__max_depth': max_depth,
                'kc__min_samples_split': min_samples_split,
                'kc__min_samples_leaf': min_samples_leaf,
                'kc__bootstrap': bootstrap}
'''
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
'''

rf_validator = RandomizedSearchCV(estimator = rf_pipeline,
								  cv=2,
								  param_distributions=random_grid,
								  n_iter = 5,
								  verbose=0,
								  random_state=42,
								  n_jobs=-1)

rf_validator.fit(X_train, y_train)
aapl_reg.get_results(rf_validator, X_test, y_test, encoder)











#sample_weights = compute_sample_weight('balanced', y_train)
#rf_validator.fit(X_train, y_train, sample_weight=sample_weights)



print('The parameters of the best model are: ')
        print(model.best_params_)
        results = model.cv_results_

        best_model = model.best_estimator_
        print(best_model)

        yhat = best_model.predict(X_test)
        yhat_proba = best_model.predict_proba(X_test)

        yhat_inv = self.decode_yhat(yhat, yhat_proba, encoder)

        y_inv = self.decode_y(y_test, encoder)
        df_inv = pd.concat([yhat_inv, y_inv], axis=1)

        df_inv['rank'] = df_inv.groupby(['y_label'])['yhat_proba'].transform(
            lambda x: pd.qcut(x, 5, labels=range(1, 6)))

        df_inv['rank'] = df_inv['rank'].astype('int32')

        print(
            'Balanced Accuracy Score (Overall): \n' + str(
                balanced_accuracy_score(df_inv['y_label'], df_inv['yhat_label'])))
        print('Balanced Crosstab Rank (Overall): \n' + str(
            pd.crosstab(df_inv['y_label'], df_inv['yhat_label'], rownames=['Actual'], colnames=['Predicted'])))

        rcount = list(df_inv['rank'].unique())
        for i in range(1, len(rcount) + 1):
            df = df_inv[df_inv['rank'] == i]
            print('Balanced Accuracy Score Rank \n' + str(i) + ' ' + str(
                balanced_accuracy_score(df['y_label'], df['yhat_label'])))
            print('Balanced Crosstab Rank \n' + str(i) + ' ' + str(
                pd.crosstab(df['y_label'], df['yhat_label'], rownames=['Actual'], colnames=['Predicted'])))













#Now re-fit the classifier using the best parameters and against all of the data
#Make this as up to date as possible with data

final_rf_classifier = RandomForestClassifier(bootstrap=False, criterion='entropy', max_depth=110,
                       max_features='sqrt', min_samples_leaf=2,
                       n_estimators=600, random_state=42,
                        class_weight='balanced')


final_rf_pipeline = Pipeline([
    ('scaler',RobustScaler()),
	('pca', PCA(n_components=0.97)),
    ('kc', final_rf_classifier)
])



final_rf_pipeline.fit(X_classical, encoded_y)

final_rf_pipeline.evaluate(X_classical, encoded_y)



