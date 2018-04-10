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

# Build optimal model using backward elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((X.shape[0], 1)).astype(int), values = X, axis = 1)
regressor_OLS = sm.OLS(endog = y, exog = X).fit()
while((X.size != 0) and (np.max(regressor_OLS.pvalues) > 0.05)):
    X = np.delete(X, np.argmax(regressor_OLS.pvalues), 1)
    regressor_OLS = sm.OLS(endog = y, exog = X).fit()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict
y_pred = regressor.predict(X_test)