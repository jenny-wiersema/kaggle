#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 12:56:06 2020

@author: jenny-wiersema
"""

########################
### import libraries ###
########################

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

from titanic_utils import corr_heatmap, parse_values, preprocess_features


sns.set_style('white')

#################
### load data ###
#################

## passengerid: index
## survived: Binary, 0 = No, 1 = Yes. The value to be predicted
## pclass: ticket class, {1,2,3}
## name: name of passenger
## sex: sex of passenger, {female, male}
## age: age of passenger, float
## sibsp: number of siblings/spouses on titanic. integer
## parch: number of parents/children on titanic. integer
## ticket: ticket number
## fare: passenger fare, float
## cabin: cabin number
## embarked: port of embarkation, C = Cherbourg, Q = Queenstown, S = Southampton


## training ##
dat = pd.read_csv('train.csv', index_col = 'PassengerId')
dat.columns = [c.lower() for c in dat.columns] ## set all column names to lowercase

y = dat['survived']
dat = dat.drop(columns = 'survived')

## testing ##
dat_test = pd.read_csv('test.csv', index_col = 'PassengerId')
dat_test.columns = [c.lower() for c in dat_test.columns] ## set all column names to lowercase


###########################
### initial data review ###
###########################

dat.info()

## review numeric features ##

dat_summary = dat.describe()

dat.hist(bins=50, figsize=(20,15))

corr_heatmap(dat.corr())

scatter_matrix(dat[dat_summary.columns], figsize=(8, 8))

## review non numeric features ##

cols = list(set(dat.columns) - set(dat_summary.columns))

for c in cols:
    print('\n' + c.upper() + '\n')
    print(dat[c].value_counts())
    input('Press Enter to continue...')

    
#######################
### preprocess data ###
#######################

dat_features = preprocess_features(dat)
    
###########################
### logistic regression ###
###########################

## calculate regression ##

log_reg = LogisticRegression()

scores = cross_val_score(log_reg, dat_features, y, cv = 10, scoring = 'f1')

scores.mean()

log_reg.fit(dat_features, y)

## prediction ##

y_test = log_reg.predict(preprocess_features(dat_test))

output = pd.DataFrame(y_test, index = dat_test.index, columns = ['Survived'])
output.index.name = 'PassengerId'
output.to_csv('candidate_logistic.csv')

## review features ##

XS_comp = pd.DataFrame(index = dat_features.columns)
XS_comp['std'] = X.std()

XS_comp['coef_log'] = log_reg.coef_.T


######################
### SGD Classifier ###
######################

## run regression ##

sgd_clf = SGDClassifier(random_state = 42)

sgd_clf.fit(dat_features, y)

## predict test values ##

y_sgd = sgd_clf.predict(preprocess_features(dat_test))

output = pd.DataFrame(y_sgd, index = dat_test.index, columns = ['Survived'])
output.index.name = 'PassengerId'
output.to_csv('candidate_sgd.csv')

XS_comp['coef_sgd'] = sgd_clf.coef_.Tg


## validate results ##

y_scores = cross_val_predict(sgd_clf, dat_features, y, cv=3,
                             method = 'decision_function')


precisions, recalls, thresholds = precision_recall_curve(y, y_scores)

plt.figure(figsize=(8, 4))
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.xlim([-10, 10])
plt.show()

plt.figure(figsize = (4,4))
plt.plot(recalls, precisions)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.axis([0, 1, 0, 1])
plt.show()


