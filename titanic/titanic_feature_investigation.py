#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 13:35:55 2020

@author: jenny-wiersema
"""

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

dat.hist(bins=50, figsize=(6,6))

corr_heatmap(dat.corr(), save_plot = True, plot_file = 'data_heatmap')

scatter_matrix(dat[dat_summary.columns], figsize=(8, 8))
plt.savefig('data_scatterplot.png', bbox_inches = 'tight')

## review non numeric features ##

cols = list(set(dat.columns) - set(dat_summary.columns))

for c in cols:
    print('\n' + c.upper() + '\n')
    print(dat[c].value_counts())
    input('Press Enter to continue...')
    
    
###########################
## Feature Investigation ##
###########################

    
############    
## pclass ##
############
    
survival_plot(dat['pclass'], y, plot_title = 'Ticket Class', save_plot = True, 
         plot_file = 'pclass_survival')

pclass_linear = linear_review(dat['pclass'], y)

##########
## name ##
##########

titles = ['Miss.', 'Mr.', 'Mrs', 'Master.', 'Rev.', 'Dr.', 'Ms.', 'Mme.','Don.',
          'Major.','Sir.','Mlle.','Col.', 'Capt.', 'Countess', 'Jonkheer.']

parsed_name = name_analysis(dat[['name']], titles)


common_titles = list(parsed_name.value_counts().loc[parsed_name.value_counts() > 5].index.values)
uncommon_titles = list(parsed_name.value_counts().loc[parsed_name.value_counts() <= 5].index.values)

I1 = [i in common_titles for i in parsed_name]
I2 = [i in uncommon_titles for i in parsed_name]

survival_plot(parsed_name[I1], y[I1], plot_title = 'Common Passenger Titles', 
              save_plot = True, plot_file = 'name_survival1', figsize = (8,3))

survival_plot(parsed_name[I2], y[I2], plot_title = 'Uncommon Passenger Titles', 
              save_plot = True, plot_file = 'name_survival2', figsize = (12,3))

#########
## sex ##
#########

survival_plot(dat['sex'], y, plot_title = 'Passenger Sex', save_plot = True, 
              plot_file = 'sex_survival')

X = dat['sex'] + ' - class ' + [str(i) for i in dat['pclass']]

survival_plot(X, y, plot_title = 'Passenger Sex by Class', figsize = (14,5), 
              save_plot = True, plot_file = 'sex_class_survival')

#########
## age ##
#########

plot_hist(dat['age'], save_plot = True, plot_file = 'age_dist')


age_linear = linear_review(dat['age'], y)

I = ~dat['age'].isnull()

X = bin_analysis(dat[['age']].loc[I], n_bins = 8, strategy = 'uniform')
X.name = 'age_all'

xticks = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80']

survival_plot(X, y, plot_title = 'Passenger Ages', figsize = (10,4), 
              xticks = xticks, save_plot = True, plot_file = 'age_survival')

## separate by gender

I1 = dat['sex'].loc[I] == 'female'
I1 = I1.loc[I1]

X1 = X.loc[I1.index]
X1.name = 'age_female'


I2 = dat['sex'].loc[I] == 'male'
I2 = I2.loc[I2]

X2 = X.loc[I2.index]
X2.name = 'age_male'


X_all = pd.concat([X, X1, X2], axis = 1)

survival_plot(X_all, y, plot_title = 'Passenger Ages by Gender', 
              figsize = (10,4), save_plot = True, 
              plot_file = 'age_gender_survival', xticks = xticks)


###########
## sibsp ##
###########

survival_plot(dat['sibsp'], y, plot_title = 'Siblings/Spouses', save_plot = True, 
              plot_file = 'sibsp_survival', figsize = (6,3))

I1 = dat['sex'] == 'female'
I2 = dat['sex'] == 'male'


X_all = pd.concat([dat['sibsp'], dat['sibsp'].loc[I1], dat['sibsp'].loc[I2]], axis = 1)
X_all.columns = ['sibsp_all', 'sibsp_female', 'sibsp_male']

survival_plot(X_all, y, plot_title = 'Passenger Siblings/Spouses by Gender', 
              figsize = (6,3), save_plot = True, 
              plot_file = 'sibsp_gender_survival')

###########
## parch ##
###########

survival_plot(dat['parch'], y, plot_title = 'Parents/Children', save_plot = True, 
              plot_file = 'parch_survival', figsize = (6,3))


I1 = dat['sex'] == 'female'
I2 = dat['sex'] == 'male'


X_all = pd.concat([dat['parch'], dat['parch'].loc[I1], dat['parch'].loc[I2]], axis = 1)
X_all.columns = ['parch_all', 'parch_female', 'parch_male']

survival_plot(X_all, y, plot_title = 'Passenger Siblings/Spouses by Gender', 
              figsize = (6,3), save_plot = True, 
              plot_file = 'parch_gender_survival')



I1 = dat['pclass'] == 1
I2 = dat['pclass'] == 2
I3 = dat['pclass'] == 3


X_all = pd.concat([dat['parch'], dat['parch'].loc[I1], dat['parch'].loc[I2],
                  dat['parch'].loc[I3]],  axis = 1)
X_all.columns = ['parch_all', 'parch_class1', 'parch_class2', 'parch_class3']

survival_plot(X_all, y, plot_title = 'Passenger Siblings/Spouses by Passenger Class', 
              figsize = (6,3), save_plot = True, 
              plot_file = 'parch_pclass_survival')

############
## ticket ##
############

None

##########
## fare ##
##########

dat[['fare']].describe()

plot_hist(dat['fare'], save_plot = True, plot_file = 'fare_dist')

fare_linear = linear_review(dat['fare'], y)
fare_linear_clipped = linear_review(dat['fare'].clip(0,100), y)


X = bin_analysis(dat[['fare']], n_bins = 10, strategy = 'quantile')
X.name = 'fare_all'


survival_plot(X, y, plot_title = 'Passenger Fares', figsize = (10,4),
              save_plot = True, plot_file = 'fare_survival')

## separate by gender

I1 = dat['sex'] == 'female'
I1 = I1.loc[I1]

X1 = X.loc[I1.index]
X1.name = 'fare_female'


I2 = dat['sex'] == 'male'
I2 = I2.loc[I2]

X2 = X.loc[I2.index]
X2.name = 'fare_male'


X_all = pd.concat([X, X1, X2], axis = 1)

survival_plot(X_all, y, plot_title = 'Passenger Fares by Gender', 
              figsize = (10,4), save_plot = True, 
              plot_file = 'fare_gender_survival')


###########
## cabin ##
###########

cabins = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T']

I = ~dat['cabin'].isnull()


# cabin or no cabin

X = pd.DataFrame('no cabin', index = dat.index, columns = ['cabin'])
X.loc[I] = 'cabin'

survival_plot(X, y, plot_title = 'Cabin Defined', 
              save_plot = True, plot_file = 'cabin_survival1', figsize = (8,3))


I1 = dat['sex'] == 'female'
I2 = dat['sex'] == 'male'

X_all = pd.concat([X, X.loc[I1], X.loc[I2]], axis = 1)
X_all.columns = ['cabin_all', 'cabin_female', 'cabin_male']


survival_plot(X_all, y, plot_title = 'Cabin Separated by Gender', 
              save_plot = True, plot_file = 'cabin_survival1_gender', figsize = (8,3))


I1 = dat['pclass'] == 1
I2 = dat['pclass'] == 2
I3 = dat['pclass'] == 3

X_all = pd.concat([X, X.loc[I1], X.loc[I2], X.loc[I3]], axis = 1)
X_all.columns = ['cabin_all', 'cabin_class1', 'cabin_class2', 'cabin_class3']


survival_plot(X_all, y, plot_title = 'Cabin Separated by Passenger Class', 
              save_plot = True, plot_file = 'cabin_survival1_pclass', figsize = (8,3))


# Deck Level #

parsed_cabins = name_analysis(dat[['cabin']].loc[I], cabins)
 
survival_plot(parsed_cabins, y, plot_title = 'Cabin Floor', 
              save_plot = True, plot_file = 'cabin_survival1', figsize = (8,3), 
              xticks = cabins)

I1 = dat['sex'].loc[I] == 'female'
I2 = dat['sex'].loc[I] == 'male'



X_all = pd.concat([parsed_cabins, parsed_cabins.loc[I1], parsed_cabins.loc[I2]], axis = 1)
X_all.columns = ['cabin_all', 'cabin_female', 'cabin_male']

survival_plot(X_all, y, plot_title = 'Deck Level by Gender', 
              figsize = (6,3), save_plot = True, 
              plot_file = 'cabin_gender_survival', xticks = cabins)



# number of cabins assigned #

n_cabins = pd.DataFrame([len(x.split(' ')) for x in X], index = X.index, columns = ['n_cabin'])


survival_plot(n_cabins, y, plot_title = 'Number of Cabins', 
              figsize = (6,3), save_plot = True, 
              plot_file = 'cabin_gender_survival')


# odd or even cabin #

I = ~dat['cabin'].isnull()
x = dat['cabin'].loc[I]

I2 = [len(i)>1 for i in x]

x = x.loc[I2]

X = pd.DataFrame([int(i[-1])%2 for i in x], index = x.index, columns = ['side'])


survival_plot(X, y, plot_title = 'Cabin by Side of Boat', 
              figsize = (6,3), save_plot = True, 
              plot_file = 'cabin_side_survival', xticks = ['Starboard', 'Port'])


I1 = dat['sex'] == 'female'
I2 = dat['sex'] == 'male'

X_all = pd.concat([X, X.loc[I1], X.loc[I2]], axis = 1)
X_all.columns = ['cabin_all', 'cabin_female', 'cabin_male']


survival_plot(X_all, y, plot_title = 'Cabin by Side of Boat', 
              figsize = (6,3), save_plot = True, 
              plot_file = 'cabin_side_gender_survival', xticks = ['Starboard', 'Port'])



#################
## embarkation ##
#################

survival_plot(dat['embarked'], y, plot_title = 'Port of Embarkation', save_plot = True, 
         plot_file = 'embarked_survival')



