
import random
import datetime
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, RandomForestClassifier
from sklearn.svm import SVC,SVR,LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn import pipeline, grid_search
from sklearn.decomposition import TruncatedSVD
from sklearn import decomposition
from sklearn import preprocessing
from sklearn import tree
# from sklearn.lda import LDA

from sklearn import neighbors

random.seed(1729)
start_time = time.time()
date = str(datetime.datetime.now().strftime(format='%m%d'))

print("::Start time- ", datetime.datetime.now())

train = pd.read_csv('Train.csv', encoding="ISO-8859-1")#[:1000]
test = pd.read_csv('Test.csv', encoding="ISO-8859-1")#[:1000]

train_count = train.shape[0]

train_test = pd.concat((train, test), axis=0,ignore_index=True)

train_test['Cust_status'] = train_test['Cust_status'].map(lambda x: 2 if x=='Old' else 1)
train_test['Trans24'] = train_test['Trans24'].map(lambda x: 2 if x == 'Enable' else 1)
train_test['Trans25'] = train_test['Trans25'].map(lambda x: 2 if x == 'Enable' else 1)
train_test['Trans26'] = train_test['Trans26'].map(lambda x: 2 if x == 'Enable' else 1)
train_test['Trans27'] = train_test['Trans27'].map(lambda x: 2 if x == 'Enable' else 1)

cols = list(train.columns.values)


cols.remove('Cust_id')
cols.remove('Active_Customer')



for i in cols:
    train_test[i] = train_test[i].fillna(train_test[i].mean(axis=0))


trans = [col for col in cols if 'Trans' in col]

food = [col for col in cols if 'Food' in col]

promotion = [col for col in cols if 'Promotion' in col]


sum = 0

for i in train_test:
    for j in trans:
        sum = sum + train_test[j]
    train_test['trans_avg'] = sum/41

sum = 0
for i in train_test:
    for j in food:
        sum = sum + train_test[j]
    train_test['trans_avg'] = sum/164

sum = 0
for i in train_test:
    for j in promotion:
        sum = sum + train_test[j]
    train_test['trans_avg'] = sum/48

train_test = train_test.drop(np.concatenate((trans, food, promotion),axis=0).ravel(), axis=1)

print(train_test.shape)

test_id = train_test['Cust_id'][train_count:]
Y_train = train_test['Active_Customer'][:train_count].values

train_test = train_test.drop(['Cust_id','Active_Customer'],axis=1)



X_train = train_test[:train_count]
X_test = train_test[train_count:]




# Logistic Regression - Fine
# clf = linear_model.LogisticRegression(verbose=1,n_jobs=-1)
# clf.fit(X_train,Y_train)
# b = clf.coef_
# b0 = clf.intercept_
# print(clf.score(X_train,Y_train))
# Y_pred = clf.predict(X_test)



# Random Forest Regression - Good one
clf = RandomForestRegressor(n_estimators=600, n_jobs=-1, random_state=2016, verbose=1)
clf.fit(X_train,Y_train)
Y_pred = clf.predict(X_test)

# Random Forest Classifier - less Good one
# clf = RandomForestClassifier(n_estimators=250, max_features=0.1,bootstrap=True,
#                              n_jobs=-1, random_state=2016, verbose=1, oob_score=False)
# clf.fit(X_train,Y_train)
# Y_pred = clf.predict(X_test)


# clf = neighbors.KNeighborsClassifier(1, 'uniform')
# clf.fit(X_train,Y_train)
# Y_pred = clf.predict(X_test)




logistic = linear_model.LogisticRegression(verbose=1,n_jobs=-1,solver='sag',tol=0.01)
# forest_classify = RandomForestClassifier( max_features=0.1,bootstrap=False,
#                              n_jobs=-1, random_state=2016, verbose=1, oob_score=False)
# forest_classify = RandomForestRegressor(n_jobs=-1, random_state=2016, verbose=1)
# pca = decomposition.PCA()
stdscalar = preprocessing.MinMaxScaler()
#
# n_components = [100, 150, 200, 250]
# n_estimators = [100, 200]
Cs = np.logspace(0.01,1)
# #tsvd = TruncatedSVD(n_components=200, random_state=2016)
pipe = pipeline.Pipeline([('stdscalar',stdscalar),('logistic', logistic)])
# param_grid = {'rfr__max_features': [1, 0.1, "log2"]}
# model = grid_search.GridSearchCV(pipe, dict(pca__n_components=n_components,
#                               forest_classify__n_estimators=n_estimators), n_jobs=-1, verbose=1)
model = grid_search.GridSearchCV(pipe, dict(Cs, n_jobs=-1, verbose=1), n_jobs=-1, verbose=1)
model.fit(X_train, Y_train)

print("Best parameters found by grid search:")
print(model.best_params_)
print("Best CV score:")
print(model.best_score_)
#
Y_pred = model.predict(X_test)




print(Y_pred.shape)

# x= []
# for i in Y_pred:
#     x.append(round(i))


# print(x)

# print(clf.score(X_train,Y_train))

filename = 'submission_' + date + '.csv'

# pd.DataFrame({"Cust_id":test_id,"Active_Customer":x}).to_csv(filename, index=False)

pd.DataFrame({"Cust_id":test_id,"Active_Customer":Y_pred}).to_csv(filename, index=False)






