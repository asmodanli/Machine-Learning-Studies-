# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 21:17:28 2018

@author: Sena
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

lr=LinearRegression()

df= pd.read_csv("polynomial-regression.csv", sep = ";")


y = df.araba_max_hiz.values.reshape(-1,1)
x = df.araba_fiyat.values.reshape(-1,1)

plt.scatter(x,y)
plt.xlabel("Arabanin max hizi")
plt.ylabel("Arabanin fiyati")
plt.show()

lr.fit(x,y)

y_head = lr.predict(x)



#%%

y = df.araba_max_hiz.values.reshape(-1,1)
x = df.araba_fiyat.values.reshape(-1,1)

from sklearn.preprocessing import PolynomialFeatures
polynomial_regression = PolynomialFeatures(degree = 2)
#degree = x^n yani x^2 ye kadar gidecez

x_polynomial = polynomial_regression.fit_transform(x)

#polynomial feature i kullan fiyatı 2. dereceden polinoma çevir

linear_reg2 = LinearRegression()
linear_reg2.fit(x_polynomial,y)


#%%

y_head2 = linear_reg2.predict(x_polynomial)

plt.plot(x,y_head2, color = "green", label ="pol")
plt.legend()
plt.show()























