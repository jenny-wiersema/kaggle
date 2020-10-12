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
import random
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import metrics

from utils import linear_review, sigmoid, create_features

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

dat_test = pd.read_csv('test.csv', index_col = 'PassengerId')
dat_test.columns = [c.lower() for c in dat_test.columns] ## set all column names to lowercase


#############################
### initial data overview ###
#############################

print('Percentage of Survivors: ' + str(round(dat.survived.mean(),3)*100) + '%')

## Ticket Class ##
print('')
print('Percentage of Survivors by Class:')
print(round(dat.groupby(by = 'pclass').survived.mean()*100,1))

## Name ##

titles = ['Miss.', 'Mr.', 'Mrs', 'Master.', 'Rev.', 'Dr.', 'Ms.', 'Mme.','Don.',
          'Major.','Sir.','Mlle.','Col.', 'Capt.', 'Countess', 'Jonkheer.']
passenger_titles = pd.DataFrame(index = dat.index)
dat['title'] = 'Other'

for t in titles:
    passenger_titles[t] = [int(t in d) if type(d) == str else 0 for d in dat['name']]
    dat.loc[passenger_titles[t] == 1, 'title'] = t

print('')
print('Percentage of Survivors by Title')
print(dat[['survived', 'title']].groupby(by = 'title').agg(['mean','count']))


## Sex ##
print('')
print('Percentage of Survivors by Sex:')
print(dat[['survived','sex']].groupby(by = 'sex').agg(['mean','count']))

# a passenger is more likely to survive if they are female

## Age ##

age_stats = linear_review(dat,'age')

## Siblings/Spouses ##

sibsp_stats = linear_review(dat,'sibsp')

## Parents/Children ##

parch_stats = linear_review(dat, 'parch')

## Fare ##

fare_stats = linear_review(dat, 'fare')

## Cabin ##
cabin_binary = pd.DataFrame(dat['survived'])
cabin_binary['cabin'] = [type(d) == str for d in dat['cabin']]
print('')
print('Percentage of Survivors by Cabin:')
print(round(cabin_binary.groupby(by = 'cabin').survived.mean()*100,1))

cabins = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T']
cabin_floors = pd.DataFrame(index = cabin_binary.index)
dat['cabin_floor'] = 'N/A'

for c in cabins:
    cabin_floors[c] = [int(c in d) if type(d) == str else 0 for d in dat['cabin']]
    dat.loc[cabin_floors[c] == 1, 'cabin_floor'] = c

print('')
print('Percentage of Survivors by Cabin Floor')
print(round(dat.groupby(by = 'cabin_floor').survived.mean()*100,1))

## Embarkation Location ##

print('')
print('Percentage of Survivors by Embarkation Location:')
print(round(dat.groupby(by = 'embarked').survived.mean()*100,1))

## Overall ##
stats_all = age_stats.append(sibsp_stats).append(parch_stats).append(fare_stats)
print('')
print('Regression Statistics')
print(round(stats_all,3))


'''
What I learned:
    - Survival is 38.4%
    - Survival is higher for 1st or 2nd class passengers
    - Women are 3-4 times more likely to survive than men
    - People embarking in Cherbourg are more likely to survive
    - Younger people are more likely to survive than older, but t-stats ~= 2
    - number of siblings/spouses has t-stat of ~1, so not very much information linearly
    - more parents/children on board mean a higher likelihood of surviving
    - having a cabin means you're twice as likely to survive
    - a higher fare has a much higher likelihood of survival 
    - no Rev. survived
'''

   
###########################
### Logistic Regression ###
###########################

train_features = ['sex','pclass','name','age','sibsp','parch','fare','cabin','embark']

y = dat['survived']


### model selection - regularization ###

# subset data into test & train
random.seed(0)
I = np.arange(dat.shape[0])
random.shuffle(I)
test_size = int(np.floor([dat.shape[0]*0.2]))

C_check = [0.03, 0.1, 0.3, 1, 3, 10, 30] ## inverse of regularization parameter
xtick_labels = [str(c) for c in C_check]
xtick_labels.insert(0,'0')
score_reg = pd.DataFrame(index = C_check, columns = ['train','test'])


all_features = create_features(dat, train_features)

for c in C_check:
    log_reg = LogisticRegression(C = c)
    score_train = []
    score_test = []
    for i in np.arange(5):
        ## subset data into testing and training data, using K-folds with K=5
        I_test = I[test_size*i:test_size*(i+1)]
        I_test.sort()
        I_train = np.setdiff1d(I, I_test)

        ## scale data
        X_train = all_features.iloc[I_train]
        mu = X_train.mean()
        sigma = X_train.std()
        sigma[sigma == 0] = 1 #avoid dividing by 0 for cases where there is no stdev
        X_train = (X_train - mu)/sigma
        X_test = (all_features.iloc[I_test] - mu)/sigma
        
        ## run regression
        log_reg.fit(X_train, y.iloc[I_train])
        score_train.append(log_reg.score(X_train, y.iloc[I_train]))
        score_test.append(log_reg.score(X_test, y.iloc[I_test]))
        
    score_reg.loc[c,'train'] = np.array(score_train).mean()
    score_reg.loc[c,'test']  = np.array(score_test).mean()
        
fig, ax = plt.subplots()
ax.plot(np.arange(len(C_check)), score_reg[['train','test']])
ax.set(title='Score by Regularization Parameter', 
       xlabel = 'Regularization Parameter',
       xticklabels = xtick_labels)
ax.legend(['Train','Test'])

''' Both test & train scores are fairly insensitive to regularization parameter '''

### predict test model ###


## scale data
X_train = create_features(dat, train_features)
mu = X_train.mean()
sigma = X_train.std()
sigma[sigma == 0] = 1 #avoid dividing by 0 for cases where there is no stdev
X_train = (X_train - mu)/sigma
X_test = create_features(dat_test, train_features)
X_test = (X_test[X_train.columns] - mu)/sigma

## run regression
log_reg = LogisticRegression(C = 0.1)
log_reg.fit(X_train, y)

y_output = pd.DataFrame(log_reg.predict(X_test), index = X_test.index, 
                        columns = ['Survived'])
y_output.to_csv('candidate.csv')


##############################
### Evalulate Coefficients ###
##############################

coef = pd.DataFrame(log_reg.coef_.transpose(), index = X_train.columns, columns = ['coef'])
coef.plot.bar()

######################
### Error Analysis ###
######################

## identify which passengers have an error

y_train = log_reg.predict(X_train)
I_err = y != y_train

y_err = pd.DataFrame(log_reg.predict_proba(X_train), index = X_train.index)
y_err = y_err.loc[I_err]



###################################
### Evalulate removing features ###
###################################

train_features = ['sex','pclass','name','age','sibsp','parch','fare','cabin','embark']
reg_param = 0.1
scores_feat = pd.DataFrame(index = train_features, columns = 
                          ['only1_train','only1_test','minus1_train', 'minus1_test'])
y = dat['survived']


# subset data into test & train
random.seed(0)
I = np.arange(dat.shape[0])
random.shuffle(I)
test_size = int(np.floor([dat.shape[0]*0.2]))


for f in train_features:
    # logistic regression, only have 1 feature
    feat = create_features(dat, f)
    log_reg = LogisticRegression(C = reg_param)
    score_train = []
    score_test = []
    for i in np.arange(5):
        ## subset data into testing and training data, using K-folds with K=5
        I_test = I[test_size*i:test_size*(i+1)]
        I_test.sort()
        I_train = np.setdiff1d(I, I_test)

        ## scale data
        X_train = feat.iloc[I_train]
        mu = X_train.mean()
        sigma = X_train.std()
        sigma[sigma == 0] = 1 #avoid dividing by 0 for cases where there is no stdev
        X_train = (X_train - mu)/sigma
        X_test = (feat.iloc[I_test] - mu)/sigma
        
        ## run regression
        log_reg.fit(X_train, y.iloc[I_train])
        score_train.append(log_reg.score(X_train, y.iloc[I_train]))
        score_test.append(log_reg.score(X_test, y.iloc[I_test]))
        
    scores_feat.loc[f,'only1_train'] = np.array(score_train).mean()
    scores_feat.loc[f,'only1_test']  = np.array(score_test).mean()
        
    # logistic regression, all but 1 feature
    feat_list = train_features[:]
    feat_list.remove(f)
    feat = create_features(dat, feat_list)
    log_reg = LogisticRegression(C = reg_param)
    score_train = []
    score_test = []
    for i in np.arange(5):
        ## subset data into testing and training data, using K-folds with K=5
        I_test = I[test_size*i:test_size*(i+1)]
        I_test.sort()
        I_train = np.setdiff1d(I, I_test)

        ## scale data
        X_train = feat.iloc[I_train]
        mu = X_train.mean()
        sigma = X_train.std()
        sigma[sigma == 0] = 1 #avoid dividing by 0 for cases where there is no stdev
        X_train = (X_train - mu)/sigma
        X_test = (feat.iloc[I_test] - mu)/sigma
        
        ## run regression
        log_reg.fit(X_train, y.iloc[I_train])
        score_train.append(log_reg.score(X_train, y.iloc[I_train]))
        score_test.append(log_reg.score(X_test, y.iloc[I_test]))
        
    scores_feat.loc[f,'minus1_train'] = np.array(score_train).mean()
    scores_feat.loc[f,'minus1_test']  = np.array(score_test).mean()
    
    
fig, ax = plt.subplots()
ax.plot(np.arange(len(C_check)), score_reg[['train','test']])
ax.set(title='Score by Regularization Parameter', 
       xlabel = 'Regularization Parameter',
       xticklabels = xtick_labels)
ax.legend(['Train','Test'])




    