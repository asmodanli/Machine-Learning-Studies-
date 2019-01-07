# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 20:01:49 2018

@author: Sena

"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


df = pd.read_csv("multiple-linear-regression-dataset.csv",sep = ";")

x=df.iloc[:,[0,2]].values
y=df.maas.values.reshape(-1,1)

multiple_linear_regression = LinearRegression()
multiple_linear_regression.fit(x,y)

b0 = multiple_linear_regression.intercept_

b1b2 = multiple_linear_regression.coef_
 
