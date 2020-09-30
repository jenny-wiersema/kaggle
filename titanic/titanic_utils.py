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
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

sns.set_style('white')

###########################
### correlation heatmap ###
###########################

def corr_heatmap(corr):
    # generate mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    # set up matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    # generate a custom diverging colormap
    red_blue = sns.diverging_palette(230, 20, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=red_blue, vmax=1, vmin=-1, center=0,
                square=True, linewidths=1, cbar_kws={"shrink": .5})


##########################
### feature processing ###
##########################

## classes ##

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


class cap_value(BaseEstimator, TransformerMixin):
    def __init__(self, max_value = 2): # no *args or **kwargs
        self.max_value = max_value
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):        
        X[X > self.max_value] = self.max_value
        return X




def preprocess_features(dat):
    
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
    
    # imputer_only = Pipeline([
    #     ('imputer', SimpleImputer(strategy = 'median')),
    #     ])
    
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




                






        

