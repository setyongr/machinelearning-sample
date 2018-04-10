#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 01:23:27 2018

@author: yo
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')

# Get independent variable with dummy varibale for categorical data
# Also enable drop fist to avoiding dummy variable trap 
X = pd.get_dummies(dataset.iloc[:, :-1], columns=['State'], drop_first=True).values
# Get dependent variable
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predic
y_pred = regressor.predict(X_test)

# Build optimal model using backward elimination
import statsmodels.formula.api as sm
X_train = np.append(arr = np.ones((X_train.shape[0], 1)).astype(int), values = X_train, axis = 1)
regressor_OLS = sm.OLS(endog = y_train, exog = X_train).fit()
X_col = np.arange(X_train.shape[1])
while((X_train.size != 0) and (np.max(regressor_OLS.pvalues) > 0.05)):
    del_index = np.argmax(regressor_OLS.pvalues)
    X_train = np.delete(X_train, del_index, 1)
    X_col = np.delete(X_col, del_index)
    regressor_OLS = sm.OLS(endog = y_train, exog = X_train).fit()

# Predict with backwarded elimination independent variable
X_test_all = np.append(arr = np.ones((X_test.shape[0], 1)).astype(int), values = X_test, axis = 1)
X_test = np.empty((X_test_all.shape[0],0))
for i in X_col:
    X_test = np.append(arr = X_test, values = X_test_all[:, i].reshape(X_test_all.shape[0], 1), axis = 1)


y_pred_2 = regressor_OLS.predict(X_test)
