#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 20:44:34 2019

@author: Srikanth Avadhanula
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values

#importing the data
from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size = 0.3,random_state = 0)

#Here we are using LinearRegression mmodel to solve the problem
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, Y_train)

Y_pred = regression.predict(X_test)

#Visualising the Training set 
plt.scatter(X_train , Y_train , color = 'red')
plt.plot(X_train, regression.predict(X_train) , color = 'blue')
plt.title('Salary Vs Experience (Training Data)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()


#Visualising the test set
plt.scatter(X_test , Y_test , color = 'red')
plt.plot(X_train , regression.predict(X_train) , color = 'blue')
plt.title('Salary Vs Experience (Test Data)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()