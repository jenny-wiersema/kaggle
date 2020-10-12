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
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.svm import SVC

from titanic_utils import cm_dataframe, corr_heatmap, plot_ROC, precision_recall, \
    preprocess_features, review_model \

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
## embarked: port of embarkation, C = Cherbourg, Q = Queenstown, S = Southamcpton


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

## hyperparameter search ##

log_reg = LogisticRegression(max_iter=1000)

log_hparams = [{'C':[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]}]  # hyperparameters to tune

log_grid_search = GridSearchCV(log_reg, log_hparams, cv = 5, scoring = 'f1')

log_grid_search.fit(dat_features, y)

log_reg = log_grid_search.best_estimator_

log_reg.fit(dat_features, y)

## prediction ##

y_log = log_reg.predict(preprocess_features(dat_test))

output = pd.DataFrame(y_log, index = dat_test.index, columns = ['Survived'])
output.index.name = 'PassengerId'
output.to_csv('candidate_logistic.csv')

## review features ##

XS_comp = pd.DataFrame(index = dat_features.columns)
XS_comp['std'] = dat_features.std()

XS_comp['coef_log'] = log_reg.coef_.T
XS_comp['XS_log'] =  XS_comp['std']*XS_comp['coef_log']


cm_log = review_model(log_reg, dat_features, y, 'Log Reg')


#################################
### support vector classifier ###
#################################

## hyperparameter search ##

svm_clf = SVC()

svm_hparams = [{'C':[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]},
               {'kernel': ['linear','poly','rbf']}]  # hyperparameters to tune

svm_grid_search = GridSearchCV(svm_clf, svm_hparams, cv = 5, scoring = 'f1')

svm_grid_search.fit(dat_features, y)

svm_clf = svm_grid_search.best_estimator_

svm_clf.fit(dat_features, y)

## prediction ##

y_svm = svm_clf.predict(preprocess_features(dat_test))

output = pd.DataFrame(y_svm, index = dat_test.index, columns = ['Survived'])
output.index.name = 'PassengerId'
output.to_csv('candidate_svm.csv')

             
cm_svm = review_model(svm_clf, dat_features, y, 'SVM')

######################
### SGD classifier ###
######################

## run regression ##

sgd_clf = SGDClassifier(random_state = 42)

sgd_hparams = [{'alpha':np.logspace(-5,1,10)}]

sgd_grid_search = GridSearchCV(sgd_clf, sgd_hparams, cv = 5, scoring = 'f1')

sgd_grid_search.fit(dat_features, y)

sgd_clf = sgd_grid_search.best_estimator_

sgd_clf.fit(dat_features, y)

## predict test values ##

y_sgd = sgd_clf.predict(preprocess_features(dat_test))

output = pd.DataFrame(y_sgd, index = dat_test.index, columns = ['Survived'])
output.index.name = 'PassengerId'
output.to_csv('candidate_sgd.csv')

XS_comp['coef_sgd'] = sgd_clf.coef_.T
XS_comp['XS_sgd'] =  XS_comp['std']*XS_comp['coef_sgd']


## validate results ##
cm_sgd = review_model(sgd_clf, dat_features, y, 'SGD')

#####################
### Random Forest ###
#####################

forest_clf = RandomForestClassifier(n_estimators=100,random_state = 42)

forest_hparams = [{'n_estimators':[10, 30, 100, 300, 1000]}]

forest_grid_search = GridSearchCV(forest_clf, forest_hparams, cv = 5, scoring = 'f1')

forest_grid_search.fit(dat_features, y)

forest_clf = forest_grid_search.best_estimator_

forest_clf.fit(dat_features, y)

y_forest = forest_clf.predict(preprocess_features(dat_test))

output = pd.DataFrame(y_forest, index = dat_test.index, columns = ['Survived'])
output.index.name = 'PassengerId'
output.to_csv('candidate_forest.csv')


cm_forest = review_model(forest_clf, dat_features, y, 'Random Forest')


#######################
### Ensemble Method ###
#######################

## HARD ##

voting_clf_hard = VotingClassifier(
    estimators = [('logistic', log_reg), ('svm', svm_clf), ('sgd', sgd_clf), 
                  ('random forest', forest_clf)], 
    voting = 'hard')

voting_clf_hard.fit(dat_features, y)

y_ensemble_hard = voting_clf_hard.predict(preprocess_features(dat_test))


cm_ensemble_hard = cm_dataframe(y, voting_clf_hard.predict(dat_features))

output = pd.DataFrame(y_ensemble_hard, index = dat_test.index, columns = ['Survived'])
output.index.name = 'PassengerId'
output.to_csv('candidate_ensemble_hard.csv')

## SOFT ##

voting_clf_soft = VotingClassifier(
    estimators = [('logistic', log_reg), ('random forest', forest_clf)], 
    voting = 'soft')

voting_clf_soft.fit(dat_features, y)

y_ensemble_soft = voting_clf_soft.predict(preprocess_features(dat_test))

cm_ensemble_soft = cm_dataframe(y, voting_clf_soft.predict(dat_features))

output = pd.DataFrame(y_ensemble_soft, index = dat_test.index, columns = ['Survived'])
output.index.name = 'PassengerId'
output.to_csv('candidate_ensemble_soft.csv')
