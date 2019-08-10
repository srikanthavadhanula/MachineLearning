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
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean',verbose=0)
imputer = imputer.fit(X.iloc[:, 1:3])
X.iloc[:, 1:3] = imputer.transform(X.iloc[:, 1:3])


#Categorical Values
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [0])],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                         # Leave the rest of the columns untouched
)

X = np.array(ct.fit_transform(X), dtype=np.float)
labelEncoder_Y = LabelEncoder()
Y = labelEncoder_Y.fit_transform(Y)

#Splitting the dataset into training and testing data
from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size = 0.2 , random_state = 0)

'''
    FeatureScaling :: In general most of the models use Euclidean model
    in which because of the features having different ranges cause problems
    Inorder to set all the features in same Range we use FeatureScaling
'''

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
