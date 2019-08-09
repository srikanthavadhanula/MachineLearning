#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 12:10:43 2019

@author: Srikanth Avadhanula
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#To import the data we use pandas read_csv
dataset = pd.read_csv('Data.csv')

#To seperate the target and dependent values 
# iloc[] takes two parametes :: left is rows and right is columns
X = dataset.iloc[:,:-1]
Y = dataset.iloc[:,3]

#Handling missing values
from sklearn.impute import SimpleImputer
#In strategy parameter you can give 'median' , 'mode' ect.., based on your model calculation
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean',verbose=0)
imputer = imputer.fit(X.iloc[:, 1:3])
X.iloc[:, 1:3] = imputer.transform(X.iloc[:, 1:3])

