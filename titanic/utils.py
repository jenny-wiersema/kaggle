#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 14:02:54 2020

@author: jenny-wiersema
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def linear_review(dat, feature):
    
    # identify non nan values
    I = ~np.isnan(dat[feature])
    n = I.sum()

    X_train = pd.DataFrame(dat[feature].loc[I])
    Y_train = pd.DataFrame(dat['survived'].loc[I])

    X_bar = X_train.mean()

    lin_reg = LinearRegression()  
    lin_reg.fit(X_train, Y_train)

    Y_pred = lin_reg.predict(X_train)
    R2 = lin_reg.score(X_train, Y_train)

    SSR = ((Y_train - Y_pred)**2).sum().values

    std_err = (SSR/(n-2))**0.5*(((X_train - X_bar)**2).sum())**-0.5
 
    # plot predicted survival vs actual survivial, by feature
    plt.figure(figsize=(6, 3))

    plt.plot(X_train, Y_pred)
    plt.plot(X_train, Y_train, '*')
    plt.title('Survival by ' + feature.capitalize())
    plt.show()
    
    # output 
    output = pd.DataFrame(index = [feature])
    output['intercept'] = lin_reg.intercept_
    output['slope'] = lin_reg.coef_ 
    output['R2'] = R2
    output['std_err'] = std_err.values
    output['t_stat'] = abs(output['slope']/output['std_err'])
    
    return output

def sigmoid(x):
    if type(x) == list:
        x = np.array(x)
    g = 1/(1+np.exp(-x))
    return g

def create_features(dat, feature_list):
    all_features = pd.DataFrame(index = dat.index)

    ## sex ##
    if 'sex' in feature_list:
        all_features['female'] = [int('female' in d) for d in dat['sex']] #(0 = male, 1 = female)

    ## passenger class ##
    if 'pclass' in feature_list:
        for i in set(dat['pclass']):
            all_features['class_' + str(i)] = [int(d == i) for d in dat['pclass']]

    ## name ##
    if 'name' in feature_list:    
        titles = ['Dr.', 'Rev.', 'Mr.', 'Miss.', 'Mrs']
        for t in titles:
            all_features[t.lower().replace('.','')] = [int(t in d) for d in dat['name']]

    ## age ## 
    if 'age' in feature_list:
        age_buckets = [0,10,20,30,40,50,80]
        for i in np.arange(len(age_buckets)-1):
            all_features['age_' + str(age_buckets[i]) + '-' + str(age_buckets[i+1])] = [int(d >= age_buckets[i] and d < age_buckets[i+1]) for d in dat['age']]

    ## siblings/spouses ##
    if 'sibsp' in feature_list:
        for i in set(dat['sibsp']):
            all_features['sibsp_' + str(i)] = [int(d == i) for d in dat['sibsp']]

    ## parents/children ##
    if 'parch' in feature_list:
        for i in set(dat['parch']):
            all_features['parch_' + str(i)] = [int(d == i) for d in dat['parch']]

    ## fare ##
    if 'fare' in feature_list:
        fare_buckets = [0, 10, 20, 50, 100, 200, 600]
        for i in np.arange(len(fare_buckets)-1):
            all_features['fare_' + str(fare_buckets[i]) + '-' + str(fare_buckets[i+1])] = [int(d >= fare_buckets[i] and d < fare_buckets[i+1]) for d in dat['fare']]

    ## cabins ##
    if 'cabin' in feature_list:
        cabins = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T']
        for c in cabins:
            all_features['cabins_'+c] = [int(c in d) if type(d) == str else 0 for d in dat['cabin']]

    ## embarkation ##
    if 'embark' in feature_list:
        embarkations = ['C', 'Q', 'S']
        for i in embarkations:
            all_features['class_' + str(i)] = [int(d == i) for d in dat['embarked']]
    
    return all_features
    