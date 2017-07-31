from pandas import read_csv
from pandas import set_option
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
names = ['cust_num', 'savg_acc' ,' curr_acc', 'invt_acc', 'trans_dy', 'card_pay' ,'mortg_ln', 'cdt_line', 'cust_age', 'cust_sex', 'cust_mar', 'cust_chd', 'cust_inc', 'cust_car', 'cust_chn']
#dataset_train = read_csv('Banking_Churn_Data.txt', names=names)
dataset_train = read_csv('Banking_Churn_Data.txt')
#dataset_test = read_csv('New_Banking.txt')
print(dataset_train.shape)
print(dataset_train.groupby('cust_chn').size())

# 数据格式转换
dataset_train['cust_sex'] = dataset_train['cust_sex'].replace(to_replace=['male', 'female'], value=[0, 1])
print(dataset_train.groupby('card_pay').size())
dataset_train['card_pay'] = dataset_train['card_pay'].replace(to_replace=['acct', 'cheq', 'giro'], value=[0, 1, 2])
dataset_train['mortg_ln'] = dataset_train['mortg_ln'].replace(to_replace=['yes', 'no'], value=[1, 0])
dataset_train['cdt_line'] = dataset_train['cdt_line'].replace(to_replace=['yes', 'no'], value=[1, 0])
dataset_train['cust_car'] = dataset_train['cust_car'].replace(to_replace=['yes', 'no'], value=[1, 0])
dataset_train['cust_mar'] = dataset_train['cust_mar'].replace(to_replace=['married', 'single'], value=[1, 0])
dataset_train['cust_chn'] = dataset_train['cust_chn'].replace(to_replace=['customer', 'invol_chn', 'vol_churn'], value=[0, 1, 2])
#print(dataset_train.head(10))
'''
# 查看数据是否miss
for key in dataset_train:
    print(dataset_train[key].isnull)

# 计算关联关系
set_option('display.line_width', 280)
print(dataset_train.corr())
print(dataset_train.describe())

# 查看数据分布
dataset_train.hist()
plt.show()

dataset_train.plot(kind='density', subplots=True, layout=(4,4), sharex=False)
plt.show()
'''

array = dataset_train.values
X = array[:, 1:14]
y = array[:, 14]

# 评估算法的基准
num_folds = 10
seed = 7
scoring = 'accuracy'

'''
models = {}
models['LR'] = LogisticRegression()
models['LDA'] = LinearDiscriminantAnalysis()
models['KNN'] = KNeighborsClassifier()
models['CART'] = DecisionTreeClassifier()
models['NB'] = GaussianNB()
models['SVM'] = SVC()

results = []
for key in models:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(models[key], X, y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    print('%s : %f (%f)' % (key, cv_results.mean(), cv_results.std()))

pipelines = {}
pipelines['ScalerLR'] = Pipeline([('Scaler', StandardScaler()), ('LR', LogisticRegression())])
pipelines['ScalerLDA'] = Pipeline([('Scaler', StandardScaler()), ('LDA', LinearDiscriminantAnalysis())])
pipelines['ScalerKNN'] = Pipeline([('Scaler', StandardScaler()), ('KNN', KNeighborsClassifier())])
pipelines['ScalerCART'] = Pipeline([('Scaler', StandardScaler()), ('CART', DecisionTreeClassifier())])
pipelines['ScalerNB'] = Pipeline([('Scaler', StandardScaler()), ('NB', GaussianNB())])
pipelines['ScalerSVM'] = Pipeline([('Scaler', StandardScaler()), ('SVM', SVC())])
results = []
for key in pipelines:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(pipelines[key], X, y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    print('%s : %f (%f)' % (key, cv_results.mean(), cv_results.std()))
'''

'''
# 调参改进算法 - CART
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
param_grid = {}
param_grid['min_samples_leaf'] = [1, 19, 21, 23, 25, 27, 29 ]
param_grid['min_samples_split'] = [2, 4, 6, 8, 10, 12, 14 ]
param_grid['splitter'] = ['best', 'random']
param_grid['criterion'] = ['gini', 'entropy']
param_grid['random_state'] = [1, 3, 5, 7, 9, 11, 13]
model = DecisionTreeClassifier()
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(X=rescaledX, y=y)

print('最优：%s 使用%s' % (grid_result.best_score_, grid_result.best_params_))
cv_results = zip(grid_result.cv_results_['mean_test_score'],
                 grid_result.cv_results_['std_test_score'],
                 grid_result.cv_results_['params'])
#for mean, std, param in cv_results:
#    print('%f (%f) with %r' % (mean, std, param))
'''

'''
ensembles = {}
ensembles['ScaledAB'] = Pipeline([('Scaler', StandardScaler()), ('AB', AdaBoostClassifier())])
ensembles['ScaledGBM'] = Pipeline([('Scaler', StandardScaler()), ('GBM', GradientBoostingClassifier())])
ensembles['ScaledRF'] = Pipeline([('Scaler', StandardScaler()), ('RFR', RandomForestClassifier())])
ensembles['ScaledET'] = Pipeline([('Scaler', StandardScaler()), ('ET', ExtraTreesClassifier())])
results = []
for key in ensembles:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(ensembles[key], X, y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    print('%s : %f (%f)' % (key, cv_results.mean(), cv_results.std()))

# 调参改进算法 - CART
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
param_grid = {'n_estimators': [10, 20, 30, 40, 50, 60, ]}
model = RandomForestClassifier()
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(X=rescaledX, y=y)
print('最优：%s 使用%s' % (grid_result.best_score_, grid_result.best_params_))

# 调参改进算法 - CART
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
param_grid = {'n_estimators': [50, 100, 150, 200, 250, 300]}
model = GradientBoostingClassifier()
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(X=rescaledX, y=y)
print('最优：%s 使用%s' % (grid_result.best_score_, grid_result.best_params_))
'''

# 调参改进算法 - CART
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
model = DecisionTreeClassifier(min_samples_leaf=23, min_samples_split=2, random_state=5)
fit = model.fit(X, y)
print(fit.feature_importances_)
print('------------')
del(dataset_train['cust_num'])
del(dataset_train['trans_dy'])
del(dataset_train['mortg_ln'])
del(dataset_train['cdt_line'])
del(dataset_train['cust_mar'])
del(dataset_train['cust_chd'])
del(dataset_train['cust_car'])
print(dataset_train.head(1))
new_X = dataset_train.values

kfold = KFold(n_splits=num_folds, random_state=seed)
cv_results = cross_val_score(model, new_X, y, cv=kfold, scoring=scoring)
print('%s : %f (%f)' % ('CART', cv_results.mean(), cv_results.std()))
