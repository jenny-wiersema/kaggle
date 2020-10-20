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

from titanic_utils import bin_analysis, cm_dataframe, corr_heatmap, linear_review, \
    name_analysis, plot_hist, plot_ROC,\
    precision_recall, preprocess_features, review_model, survival_plot

############################
### feature_construction ###
############################

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


########################
### Feature Analysis ###
########################

## using the random forest model ##


feature_groups = 3*['pclass'] + 5*['name'] + ['sex'] + 4*['age'] + 3*['sibsp'] + \
    3*['parch'] + 4*['fare'] + 8*['cabin'] + 3*['embarked']

feat_imp = pd.DataFrame(forest_clf.feature_importances_, 
                        index = [feature_groups, dat_features.columns],
                        columns = ['Importance'])

feat_imp_agg = feat_imp.sum(level = 0).sort_values(by = 'Importance', ascending = False)


## remove 3 feature groups with lowest importance and rerun analysis ##

feat_imp_agg[-1:].index

feature_subset = dat_features.copy(deep = True)

feature_subset.drop(feat_imp.loc[feat_imp_agg[-3:].index].droplevel(level = 0).index, 
                    axis = 1, inplace = True)


forest_clf2 = RandomForestClassifier(n_estimators=100,random_state = 42)

forest_hparams = [{'n_estimators':[10, 30, 100, 300, 1000]}]

forest_grid_search = GridSearchCV(forest_clf2, forest_hparams, cv = 5, scoring = 'f1')

forest_grid_search.fit(feature_subset, y)

forest_clf2 = forest_grid_search.best_estimator_

forest_clf2.fit(feature_subset, y)


cm_forest2 = review_model(forest_clf2, feature_subset, y, 'Random Forest')



######################
### Error Analysis ###
######################

## using the random forest model ##


y_predict = forest_clf.predict(dat_features)

err_idx = y != y_predict

predict_prob = forest_clf.predict_proba(dat_features)

predict_prob = pd.DataFrame(predict_prob[err_idx,:], 
                            index = dat_features.index[err_idx], 
                            columns = ['0','1'])

predict_prob['max_prob'] = predict_prob.max(axis = 1)

predict_prob.sort_values(by = 'max_prob', ascending = False, inplace = True)



n = predict_prob.index[6]

print(predict_prob.loc[n])

dat.loc[n]










