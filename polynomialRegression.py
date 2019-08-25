#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 12:51:37 2019

@author: Srikanth
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:,1:2]
Y = dataset.iloc[:,2]


"""Since there is only one variable and that to it is already encoded there is 
no need to do encoding 

Since there are only 10 observations, it is not advisable to split into test 
and train dataset """

#LinearRegression model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)

#Polyinomial Regression

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly,Y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y)

#Visualizing the polt with LinearRegression

plt.scatter(X, Y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Linear Regression (Truth OR Bluff)')
plt.xlabel('levels')
plt.ylabel('salary')
plt.show()

#Visualizing the plot with PolynomialRegression

#After plotting with degree 2/3/4, to scatter the plot for all the value 
# use the bellow approch 
X_grid = np.arange(int(np.amin(X)), int(np.amax(X)), 0.1)
X_grid = X_grid.reshape(len(X_grid),1)
# Above for all the value
 
plt.scatter(X, Y, color='red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('Polynomial Regression (Truth OR Bluff)')
plt.xlabel('levels')
plt.ylabel('salary')
plt.show()
