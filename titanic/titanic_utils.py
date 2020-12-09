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
import random

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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer
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


class cabin_side(BaseEstimator, TransformerMixin):
    def __init__(self):
        self
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        output = X.copy()
        for i, x in enumerate(X):
            if len(x[0]) > 1:
                output[i] = ['Port' if int(x[0][-1])%2 else 'Starboard']
                        
        return output
    

  
#################
### functions ###
#################

def bin_analysis(X, n_bins, strategy = 'quantile'):
    """

    Discretizes a numerical feature into a series of boolean features

    Inputs
    ----------
    X: dataframe 
        dataframe containing a feature for all instances
    n_bins: int
        integer denoting the number of bins used to dicretize the feature
    strategy : {'uniform', 'quantile', 'kmeans'}, (default='quantile')
        Strategy used to define the widths of the bins, used in KBinsDiscretizer 

        uniform
            All bins in each feature have identical widths.
        quantile
            All bins in each feature have the same number of points.
        kmeans
            Values in each bin have the same nearest center of a 1D k-means
            cluster.
        

    Outputs
    ----------
    output_X: series
        series containing the bin number associated with each passenger
        
    """    

    pipeline_bin = Pipeline([
        ('imputer', SimpleImputer(strategy = 'median')),
        ('bins', KBinsDiscretizer(n_bins = n_bins, encode = 'ordinal', strategy = strategy)),
        ('onehot', OneHotEncoder(sparse = False)),
        ])
        
    output_X = pd.DataFrame(pipeline_bin.fit_transform(X), index = X.index)
    
    output_X = pd.get_dummies(output_X).idxmax(1)
    
    output_X.name = X.columns[0]
        
    return output_X  


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


def corr_heatmap(corr, save_plot = False, plot_file = ''):
    """

    Plots a lower triangle correlation heatmap 

    Inputs
    ----------
    corr: dataframe 
        dataframe containing a correlation matrix
    save_plot: bool
        if True, saves plot
    plot_file: str
        name for plot file
    
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
    
    if save_plot == True:
        plt.savefig(plot_file+'.png', bbox_inches = 'tight')


def linear_review(X, Y):
    """

    runs a linear regression between the X and Y data
    
    Inputs
    ----------
    X: pandas series
        series containing one feature for all instances
    Y: pandas series
        survived/not survived for training data   
    
    Outputs
    ----------
    output: dataframe
        dataframe containing metrics for the linear regression between X & Y
    
    """    
    # identify non nan values
    I = ~np.isnan(X)
    n = I.sum()

    X_train = pd.DataFrame(X.loc[I])
    Y_train = pd.DataFrame(Y.loc[I])

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
    plt.title('Survival by ' + X.name.capitalize())
    plt.show()
    
    # output 
    output = pd.DataFrame(index = [X.name])
    output['intercept'] = lin_reg.intercept_
    output['slope'] = lin_reg.coef_ 
    output['R2'] = R2
    output['std_err'] = std_err.values
    output['t_stat'] = abs(output['slope']/output['std_err'])
    
    return output


def log_reg_MC(X, Y, n_features = 20, C = 1, predict = 'cross_val', X_test = None,
               seed = 42):
    """

    Parses a series of names and outputs boolean features, which denote 
    inclusion of the titles value in the name

    Inputs
    ----------
    X: dataframe 
        dataframe containing all features for the training data set
    Y: pandas series
        survived/not survived for training data  
    n_features: int, default = 20
        number of features to include in each regression
    C: float, default = 1
        regularization parameter used in LogisticRegression function
    predict: str ['cross_val', 'test']
        determines if the predicted values will be done using cross validation on the 
        training data, or using new testing data
    X_test: default = None
        if predict = 'test', required to provide a dataframe of all features for the test data set
    seed: int, default = 20
        seed

    Outputs
    ----------
    y_test: dataframe
        dataframe containing prediction of survival or not, for training dataset
        
    """ 
        
    np.random.seed(seed)
    
    output = []
    
    for i in np.arange(100):
        cols = random.sample(list(X.columns), n_features)
        
        log_reg = LogisticRegression(C = C, max_iter = 1000)
        
        if predict == 'cross_val':
            p = cross_val_predict(log_reg, X[cols], Y, cv = 5, method='predict_proba')
        elif predict == 'test':
            log_reg.fit(X[cols], Y)
            p = log_reg.predict_proba(X_test[cols])
                            
        output.append(p[:,1])
        
    
    if predict == 'cross_val':
        output_index = X.index
    elif predict == 'test':
        output_index = X_test.index
    
    y_predict = pd.DataFrame([int(i > 0.5) for i in np.array(output).mean(axis = 0)],
                             index = output_index, columns = ['survived'])
    
    return y_predict


def name_analysis(X, titles):
    """

    Parses a series of names and outputs boolean features, which denote 
    inclusion of the titles value in the name

    Inputs
    ----------
    X: dataframe 
        dataframe containing the name feature for all instances
    titles: list
        a list containing strings of all the boolean features

    Outputs
    ----------
    output_X: dataframe
        dataframe containing processed features, with a column for each new 
        boolean feature
        
    """    
    pipeline_name = Pipeline([
        ('imputer', SimpleImputer(strategy = 'constant', fill_value = 'nan')),
        ('parsing_name', parse_values(feature_list = titles)),
        ])

    output_X = pipeline_name.fit_transform(X)
    
    output_X = pd.get_dummies(output_X).idxmax(1)
    
    output_X.name = X.columns[0]
    
    output_X.index = X.index
    
    return output_X     


def plot_coefs(X, npos = 0, nneg = 0, figsize = (6,4), plot_title = '',
               save_plot = False, plot_file = ''):
    """

    Plots the coeficients with the   

    Inputs
    ----------
    X: series 
        pandas series containing a coefficents from a logistic regression
    npos: int
        number of positive coefficients to include in plot
    nneg: int
        number of negative coefficients to include in plot
    figsize: tuple
        figure size
    plot_title: str
        title for plot
    save_plot: bool
        if True, saves plot
    plot_file: str
        name for plot file
    
    Outputs
    ----------
    None
    
    """
    X = X.sort_values()
    X = pd.concat([X[:nneg], X[-npos:]])
    plt.figure(figsize = figsize)
    X.plot(kind = 'bar')
    plt.subplots_adjust(bottom = 0.3)
    plt.title(plot_title)
    plt.show()
    
    if save_plot == True:
        plt.savefig(plot_file+'.png', bbox_inches = 'tight')


def plot_hist(X, save_plot = False, plot_file = '', figsize = (6,4)):
    """

    Creates a histogram for a feature  

    Inputs
    ----------
    X: series 
        pandas series containing a feature for all instances
    save_plot: bool
        if True, saves plot
    plot_file: str
        name for plot file
    
    Outputs
    ----------
    None
    
    """
    plot_title = 'Distribution by ' + X.name.capitalize()
    plt.figure(figsize = figsize)
    sns.distplot(X)
    plt.title(plot_title)
    plt.xlabel(X.name.capitalize())
    
    if save_plot == True:
        plt.savefig(plot_file+'.png', bbox_inches = 'tight')
      

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
    
    titles = ['Dr.', 'Rev.', 'Mr.', 'Miss.', 'Mrs', 'Master']
    
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
        ('imputer', SimpleImputer(strategy = 'constant', fill_value = 'no')),
        ('parsing_name', parse_values(feature_list = cabins)),
        ])
    
    pipeline_cabin_side = Pipeline([
        ('imputer', SimpleImputer(strategy = 'constant', fill_value = 'N')),
        ('cabin_side', cabin_side()),
        ('onehot', OneHotEncoder(categories = [['Port', 'Starboard', 'N']], handle_unknown = 'ignore')),
        ])
    
    pipeline_ordinal = Pipeline([
        ('imputer', SimpleImputer(strategy = 'constant', fill_value = 'nan')),
        ('ordinal', OrdinalEncoder()),
        ])
    
    pipeline_bin = Pipeline([
        ('imputer', SimpleImputer(strategy = 'median')),
        ('bins', KBinsDiscretizer(n_bins = 5, encode = 'ordinal', strategy = 'quantile')),
        ('onehot', OneHotEncoder()),
        ])
    
    pipeline_age = Pipeline([
        ('imputer', SimpleImputer(strategy = 'median')),
        ('bins', Binarizer(threshold = 10)),    
        ])

    
    ## full pipeline ##
    
    full_pipeline = ColumnTransformer([
            ('oneshot_pclass', pipeline_onehot, ['pclass']),
            ('parsing_name', pipeline_name, ['name']),
            ('ordinal', pipeline_ordinal, ['sex']),
            ('binarizer_age', pipeline_age, ['age']),
            ('imputer_sibsp', pipeline_onehot_cap, ['sibsp']),
            ('imputer_parch', pipeline_onehot_cap, ['parch']),    
            ('bins_fare', pipeline_bin, ['fare']),
            ('parsing_cabin', pipeline_cabin, ['cabin']), 
            ('parsing_cabin_side', pipeline_cabin_side, ['cabin']),
            ('oneshot_embarked', pipeline_onehot_embarked, ['embarked']),
        ])
    

    X = full_pipeline.fit_transform(dat)     
        
    feature_names = ['pclass_' + str(i) for i in set(dat['pclass'])] \
    + ['name_' + t.lower().replace('.','') for t in titles] \
    + ['sex_male'] \
    + ['age_10+'] \
    + ['sibsp_' + i for i in ['0', '1', '2+']] \
    + ['parch_' + i for i in ['0', '1', '2+']] \
    + ['fare_q' + str(i) for i in np.arange(1,6)] \
    + ['cabin_' + c for c in cabins] \
    + ['cabin_' + c for c in ['Port', 'Starboard', 'NoCabin']] \
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



def sigmoid(x):
    if type(x) == list:
        x = np.array(x)
    g = 1/(1+np.exp(-x))
    return g
 


def survival_plot(X_all, Y, plot_title = '', save_plot = False, plot_file = '', 
                  figsize = (5,3), xticks = None):
    """

    Creates a barplot showing the probability of survival based on a feature value of a passenger

    Inputs
    ----------
    X: series 
        pandas series containing data to aggregate
    Y: pandas series
        survived/not survived for training data   
    plot_title: str
        title for plot
    save_plot: bool
        if True, saves plot
    plot_file: str
        name for plot file
    xticks: list
        Optional value, list defining xticklabels
    
    Outputs
    ----------
    None
    
    """

    # counts in each category 
    
    X_count = pd.DataFrame()
    prob_surv = pd.DataFrame()
    
    if 'Series' in str(type(X_all)):
        X_all = pd.DataFrame(X_all)
    
    
    for d in np.arange(X_all.shape[1]):
        X = X_all.iloc[:,d]
        X_count = pd.concat([X_count, X.value_counts().sort_index()], axis = 1)
 
        # probability of survivial for each category
        XY = pd.concat([X,Y.loc[X.index]], axis = 1)
        XY = XY.reset_index()
        XY = XY.set_index([X.name, 'PassengerId'])
        
        prob_surv_d = XY.mean(level = 0).sort_index()
        if X_all.shape[1] > 1:    
            prob_surv_d.rename(columns={'survived': prob_surv_d.index.name.split('_')[1]},
                               inplace = True)
            
        prob_surv_d.index.name = prob_surv_d.index.name.split('_')[0]
        prob_surv = pd.concat([prob_surv, prob_surv_d], axis = 1)
    
    prob_surv = 100*prob_surv
    
    # xticklabels
    if type(xticks) != list:
        if type(X_count.index[0]) == 'str': 
            xticks = [i.capitalize() for i in X_count.index.values]
        else:
            xticks = [i for i in X_count.index.values]
    
    xticklabels = []
    
    if X_count.shape[1] == 1:
        for i in np.arange(X_count.shape[0]):
            xticklabels.append(str(xticks[i]) + ' (' + str(X_count.values[i][0]) + ')')
    else: 
        xticklabels = xticks
    
    # plots
    fig, ax = plt.subplots(figsize = figsize)
    prob_surv.plot(kind = 'bar', ax = ax)
    
    # ax.bar(np.arange(prob_surv.shape[0]), 100*prob_surv['survived'], width = 0.6)
    plt.title(plot_title, fontsize = 14, fontweight = 3)
    plt.xlabel('')
    plt.xticks(ticks = np.arange(prob_surv.shape[0]), fontsize = 12, 
               labels = xticklabels, rotation = 0)
    plt.ylim([0, 100])
    plt.ylabel('Probability of Survivial (%)', fontsize = 12)
    if X_all.shape[1] > 1:
        plt.legend([i.capitalize() for i in list(prob_surv.columns)], fontsize = 10)
    else: 
        ax.get_legend().remove()
    plt.show()
    if save_plot == True:
        plt.savefig(plot_file+'.png', bbox_inches = 'tight')

    

    
