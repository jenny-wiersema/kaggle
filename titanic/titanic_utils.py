#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 12:55:46 2020

@author: jenny-wiersema
"""

########################
### import libraries ###
########################

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder


sns.set_style('white')

###############
### classes ###
###############

class cap_value(BaseEstimator, TransformerMixin):
    def __init__(self, max_value = 2): # no *args or **kwargs
        self.max_value = max_value
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):        
        X[X > self.max_value] = self.max_value
        return X
    
    
class parse_values(BaseEstimator, TransformerMixin):
    def __init__(self, feature_list=None): # no *args or **kwargs
        self.feature_list = feature_list
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):        
        processed = pd.DataFrame()
        for i, f in enumerate(self.feature_list):
            processed[f.lower().replace('.','')] = [int(f in d[0]) for d in X]
            
        return processed


    
#################
### functions ###
#################

def cm_dataframe(y_actual, y_predict):
    """

    Creates confusion matrix for model results
    
    Inputs
    ----------
    y_actual: array
        actual survived/not survived results
    y_predict: array
        model predicted survived/not survived results
        
    Outputs
    ----------
    cm: dataframe
        dataframe containing the confusion matrix
    
    """
    ## create dataframe from confusion matrix
    cm = confusion_matrix(y_actual, y_predict)
    
    cm = cm/cm.sum()
    
    cm = pd.DataFrame(cm, index = [0,1], columns = [0,1])

    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    
    return cm


def corr_heatmap(corr):
    """

    Plots a lower triangle correlation heatmap 

    Inputs
    ----------
    corr: dataframe 
        dataframe containing a correlation matrix containing Adj Close price data for a single date

    Outputs
    ----------
    None
    
    """
    # generate mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    # set up matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    # generate a custom diverging colormap
    red_blue = sns.diverging_palette(230, 20, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=red_blue, vmax=1, vmin=-1, center=0,
                square=True, linewidths=1, cbar_kws={"shrink": .5})


def plot_ROC(model, X, Y, model_name = ''):
    """

    plots ROC curve for model
    
    Inputs
    ----------
    model: sklearn model
        trained model 
    X: dataframe
        dataframe containing features
    Y: pandas series
        survived/not survived for training data   
    model_name: str
        model name to include in plot titles
    
    Outputs
    ----------
    None
    
    """
    
    if 'SGDClassifier' in str(type(model)):
        y_scores = cross_val_predict(model, X, Y, cv=3, method = 'decision_function')
        
    elif 'RandomForestClassifier' in str(type(model)): 
        y_prob = cross_val_predict(model, X, Y, cv=3, method = 'predict_proba')
        y_scores = y_prob[:,1]  ## score is the probability that the answer is 1
    
    else:
        y_scores = cross_val_predict(model, X, Y, cv = 3)
    fpr, tpr, thresholds = roc_curve(Y, y_scores)
    roc_score = roc_auc_score(Y, y_scores) ## area under curve
    
    plt.figure(figsize = (4,4))
    plt.plot(fpr,tpr,linewidth = 2)
    plt.plot([0,1],[0,1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(model_name + ': ROC Score: ' + str(round(roc_score,2)))
    plt.show()
    

def precision_recall(model, X, Y, model_name = ''):
    """

    Wrapper function for plotting both precision and recall vs threshold and 
    precision vs recall plots
    
    Inputs
    ----------
    model: sklearn model
        trained model 
    X: dataframe
        dataframe containing features
    Y: pandas series
        survived/not survived for training data   
    model_name: str
        model name to include in plot titles
    
    Outputs
    ----------
    None
    
    """
       
    if 'SGDClassifier' in str(type(model)):
        y_scores = cross_val_predict(model, X, Y, cv=3, method = 'decision_function')
        
    elif 'RandomForestClassifier' in str(type(model)): 
        y_prob = cross_val_predict(model, X, Y, cv=3, method = 'predict_proba')
        y_scores = y_prob[:,1]  ## score is the probability that the answer is 1
    
    else:
        y_scores = cross_val_predict(model, X, Y, cv = 3)

    precisions, recalls, thresholds = precision_recall_curve(Y, y_scores)

    plt.figure(figsize=(8, 4))
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.title(model_name)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    plt.ylim([0, 1])
    plt.show()
    
    plt.figure(figsize = (4,4))
    plt.plot(recalls, precisions)
    plt.title(model_name + ': Precision vs Recall')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.axis([0, 1, 0, 1])
    plt.show()
        
    

def preprocess_features(dat):
    """

    Plots a lower triangle correlation heatmap 

    Inputs
    ----------
    dat: dataframe 
        dataframe containing raw data for feature creation

    Outputs
    ----------
    X: dataframe
        dataframe containing processed features
    
    """
    ## feature groups ##
    
    titles = ['Dr.', 'Rev.', 'Mr.', 'Miss.', 'Mrs']
    
    cabins = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T']
    
    embarked = ['C', 'Q', 'S']
    
    ## individual pipelines ##
        
    pipeline_onehot = Pipeline([
        ('imputer', SimpleImputer(strategy = 'constant', fill_value = 0)),
        ('onehot', OneHotEncoder()),
        ])
    
    pipeline_onehot_cap = Pipeline([
        ('imputer', SimpleImputer(strategy = 'constant', fill_value = 0)),
        ('cap',cap_value(max_value = 2)),
        ('onehot', OneHotEncoder()),
        ])
        
    pipeline_onehot_embarked = Pipeline([
        ('imputer', SimpleImputer(strategy = 'constant', fill_value = 'nan')),
        ('onehot', OneHotEncoder(categories = [embarked], handle_unknown = 'ignore')),
        ])
    
    pipeline_name = Pipeline([
        ('imputer', SimpleImputer(strategy = 'constant', fill_value = 'nan')),
        ('parsing_name', parse_values(feature_list = titles)),
        ])
    
    pipeline_cabin = Pipeline([
        ('imputer', SimpleImputer(strategy = 'constant', fill_value = 'nan')),
        ('parsing_name', parse_values(feature_list = cabins)),
        ])
    
    pipeline_ordinal = Pipeline([
        ('imputer', SimpleImputer(strategy = 'constant', fill_value = 'nan')),
        ('ordinal', OrdinalEncoder()),
        ])
    
    pipeline_bin = Pipeline([
        ('imputer', SimpleImputer(strategy = 'median')),
        ('bins', KBinsDiscretizer(n_bins = 4, encode = 'ordinal', strategy = 'quantile')),
        ('onehot', OneHotEncoder()),
        ])
    
    ## full pipeline ##
    
    full_pipeline = ColumnTransformer([
            ('oneshot_pclass', pipeline_onehot, ['pclass']),
            ('parsing_name', pipeline_name, ['name']),
            ('ordinal', pipeline_ordinal, ['sex']),
            ('bins_age', pipeline_bin, ['age']),
            ('imputer_sibsp', pipeline_onehot_cap, ['sibsp']),
            ('imputer_parch', pipeline_onehot_cap, ['parch']),    
            ('bins_fare', pipeline_bin, ['fare']),
            ('parsing_cabin', pipeline_cabin, ['cabin']), 
            ('oneshot_embarked', pipeline_onehot_embarked, ['embarked']),
        ])

    X = full_pipeline.fit_transform(dat)     
        
    feature_names = ['pclass_' + str(i) for i in set(dat['pclass'])] \
    + [t.lower().replace('.','') for t in titles] \
    + ['male_female'] \
    + ['age_q' + str(i) for i in np.arange(1,5)] \
    + ['sibsp_' + i for i in ['0', '1', '2+']] \
    + ['parch_' + i for i in ['0', '1', '2+']] \
    + ['fare_q' + str(i) for i in np.arange(1,5)] \
    + ['cabin_' + c for c in cabins] \
    + ['embarked_' + i for i in embarked]
    
    
    X = pd.DataFrame(X, index = dat.index, columns = feature_names)
    
    return X

                
def review_model(model, X, Y, model_name):
    """

    wrapper function for reviewing a model's performance, including precision vs recall plots, 
    ROC plots and confusion matrix
    
    Inputs
    ----------
    model: sklearn model
        trained model 
    X: dataframe
        dataframe containing features
    Y: pandas series
        survived/not survived for training data   
    model_name: str
        model name to include in plot titles
    
    Outputs
    ----------
    cm: dataframe
        dataframe containing the confusion matrix
    
    """
    precision_recall(model, X, Y, model_name)
    
    plot_ROC(model, X, Y, model_name)
    
    cm = cm_dataframe(Y, model.predict(X))
    
    return cm
    

    plot_ROC(model, X, Y, model_name)

    cm = cm_dataframe(Y, model.predict(X))
    
    return cm
    
    

        

