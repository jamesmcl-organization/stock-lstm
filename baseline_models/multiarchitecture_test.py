import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from xgboost import XGBRegressor

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


X_train, X_test, y_train, y_test = train_test_split(X_classical, encoded_y, test_size=0.3, random_state=101)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

class_weights = compute_class_weight('balanced', np.unique(y_train), y_train)
weights_dict = dict(enumerate(class_weights))



classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    DecisionTreeClassifier(),
    XGBClassifier(learning_rate=0.02, n_estimators=600, objective='multi:softmax',silent=True, nthread=1),
    RandomForestClassifier(bootstrap=False,criterion='entropy',max_depth=20,min_samples_split=2,max_features='sqrt',
    min_samples_leaf=1,n_estimators=800,random_state=42,class_weight=weights_dict),
    AdaBoostClassifier(),
    GradientBoostingClassifier()
    ]

aapl_reg.timer(None) # timing ends here for "start_time" variable
for classifier in classifiers:
    pipe = Pipeline(steps=[('scaler',RobustScaler()),
                      ('classifier', classifier)])
    pipe.fit(X_train, y_train)
    print(classifier)
    print("model score: %.3f" % pipe.score(X_test, y_test))


RESULTS:
'''
KNeighborsClassifier(n_neighbors=3):                                    model score: 0.314
SVC(C=0.025, probability=True):                                         model score: 0.379
DecisionTreeClassifier():                                               model score: 0.370

XGBClassifier\
    (base_score=0.5, booster='gbtree', colsample_bylevel=1,
    colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
    importance_type='gain', interaction_constraints='',
    learning_rate=0.02, max_delta_step=0, max_depth=6,
    min_child_weight=1, missing=nan, monotone_constraints='()',
    n_estimators=600, n_jobs=1, nthread=1, num_parallel_tree=1,
    objective='multi:softprob', random_state=0, reg_alpha=0,
    reg_lambda=1, scale_pos_weight=None, silent=True, subsample=1,
    tree_method='exact', validate_parameters=1, verbosity=None):        model score: 0.445
RandomForestClassifier\
    (bootstrap=False,
    class_weight={0: 0.8102893890675241,
    1: 0.9097472924187726,
    2: 0.5163934426229508,
    3: 3.189873417721519, 4: 2.4},
    criterion='entropy', max_depth=20, max_features='sqrt',
    n_estimators=800, random_state=42):                                 model score: 0.580

AdaBoostClassifier():                                                   model score: 0.384
GradientBoostingClassifier():                                           model score: 0.407
'''
