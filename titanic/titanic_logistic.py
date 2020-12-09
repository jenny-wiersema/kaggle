#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 11:53:35 2020

@author: jenny-wiersema
"""

import pandas as pd
import numpy as np
import random

import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.svm import SVC

from titanic_utils import bin_analysis, cm_dataframe, corr_heatmap, linear_review, \
    log_reg_MC, name_analysis, plot_coefs, plot_hist, plot_ROC,\
    precision_recall, preprocess_features, review_model, survival_plot

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


############################
### feature_construction ###
############################

dat_features = preprocess_features(dat)

###########################
### logistic regression ###
###########################


## hyperparameter search - l1 regularization ##

log_reg_l1 = LogisticRegression(max_iter=1000, penalty = 'l1', solver = 'liblinear')

log_hparams = [{'C':[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]}]  # hyperparameters to tune

log_grid_search_l1 = GridSearchCV(log_reg_l1, log_hparams, cv = 5, scoring = 'f1')

log_grid_search_l1.fit(dat_features, y)

split_test_scores = pd.DataFrame(index = np.arange(5), columns = log_hparams[0]['C'])
split_test_scores.index.name = 'K-Fold'
split_test_scores.columns.name = 'Regularization Parameter'

i = 0
for k in log_grid_search_l1.cv_results_.keys():
    if 'split' in k:
        split_test_scores.iloc[int(k[5])] = log_grid_search_l1.cv_results_[k]


## l2 regularization ##

log_reg_l2 = LogisticRegression(max_iter=1000, penalty = 'l2')

log_grid_search_l2 = GridSearchCV(log_reg_l2, log_hparams, cv = 5, scoring = 'f1')

log_grid_search_l2.fit(dat_features, y)

split_test_scores = pd.DataFrame(index = np.arange(5), columns = log_hparams[0]['C'])
split_test_scores.index.name = 'K-Fold'
split_test_scores.columns.name = 'Regularization Parameter'

i = 0
for k in log_grid_search_l2.cv_results_.keys():
    if 'split' in k:
        split_test_scores.iloc[int(k[5])] = log_grid_search_l2.cv_results_[k]



## best model ##

# log_reg = LogisticRegression(max_iter=1000, penalty = 'l2', C = 1)
log_reg = LogisticRegression(max_iter=1000, penalty = 'l1', C = 3,  solver = 'liblinear')


log_reg.fit(dat_features, y)


## prediction ##

y_log = log_reg.predict(preprocess_features(dat_test))

output = pd.DataFrame(y_log, index = dat_test.index, columns = ['Survived'])
output.index.name = 'PassengerId'
output.to_csv('candidate_logistic_all.csv')

cm_log = review_model(log_reg, dat_features, y, 'Log Reg')

## by gender:

x0 = dat_features.loc[dat_features['sex_male'] == 0]
x1 = dat_features.loc[dat_features['sex_male'] == 1]

log_reg0 = LogisticRegression(C = 3, max_iter = 1000)
log_reg1 = LogisticRegression(C = 3, max_iter = 1000)

log_reg0.fit(x0, y.loc[x0.index])
log_reg1.fit(x1, y.loc[x1.index])

x_test = preprocess_features(dat_test)

x0_test = x_test.loc[x_test['sex_male'] == 0]
x1_test = x_test.loc[x_test['sex_male'] == 1]


y0 = pd.DataFrame(log_reg0.predict(x0_test), index = x0_test.index, columns = ['survived'])
y1 = pd.DataFrame(log_reg1.predict(x1_test), index = x1_test.index, columns = ['survived'])

y_gender = pd.concat([y0, y1], axis = 0).sort_index()
y_gender.to_csv('candidate_logistic_gender.csv')


## feature importance ##

coefs = pd.DataFrame(np.concatenate([log_reg.coef_, log_reg0.coef_, log_reg1.coef_], axis = 0).T,
                     index = dat_features.columns, columns = ['all', 'female', 'male'])

plot_coefs(coefs['all'], npos = 5, nneg = 5,
           plot_title = 'Important Features for All Passengers',
           save_plot = True, plot_file = 'coefs_all')
plot_coefs(coefs['female'], npos = 5, nneg = 5,
           plot_title = 'Important Features for Female Passengers',
           save_plot = True, plot_file = 'coefs_female')
plot_coefs(coefs['male'], npos = 5, nneg = 5,
           plot_title = 'Important Features for Male Passengers',
           save_plot = True, plot_file = 'coefs_male')
