#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 12:40:22 2020

@author: jenny-wiersema
"""

import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from pandas.plotting import scatter_matrix


## load data
dat = pd.read_csv('train.csv')

## review data
dat.info()

## review numeric features ##

dat_summary = dat.describe()

dat.hist(bins=50, figsize=(6,6))

scatter_matrix(dat[dat_summary.columns], figsize=(8, 8))


a = pd.DataFrame('ID': ['A','A','A','B','B','B'], 'score': [2,3,4,5,6,7])