#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 16:28:14 2020

@author: jenny-wiersema
"""

########################
### import libraries ###
########################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

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

dat = pd.read_csv('train.csv', index_col = 'PassengerId')
dat.columns = [c.lower() for c in dat.columns] ## set all column names to lowercase

#############################
### initial data overview ###
#############################

print('Percentage of Survivors:' + str(round(dat.survived.mean(),3)*100) + '%')

## Ticket Class ##

print('Percentage of Survivors by Class:')
print(round(dat.groupby(by = 'pclass').survived.mean()*100,1))

# a passenger is more likely to survive if they have a higher class ticket 

## Sex ##

print('Percentage of Survivors by Sex:')
print(round(dat.groupby(by = 'sex').survived.mean()*100,1))

# a passenger is more likely to survive if they are female

## Age ##

I = ~np.isnan(dat.age)

X_train = pd.DataFrame(dat['age'].loc[I])
Y_train = pd.DataFrame(dat['survived'].loc[I])
regressor = LinearRegression()  
regressor.fit(X_train, Y_train)
