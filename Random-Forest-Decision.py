# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 19:01:40 2018

@author: Sena
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("D:\DATAI\ML\decision-tree-regression-dataset.csv",sep = ";", header = None)
 
x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)


#%%

from sklearn.ensemble import RandomForestRegressor

#estimator kaç tane tree kullandığın
#*random_state = id gibi, yönteminin yani bölüş şeklinin sayısı 
#   aynı random değerlerin seçilmesini sağlar
rf =RandomForestRegressor(n_estimators = 100, random_state = 42)
rf.fit(x,y)

print("7.8 seviyesinde fiyat : ", rf.predict(7.8))

#min değerden max değere verilen değerle arttırarak git
x_ = np.arange(min(x),max(x), 0.01).reshape(-1,1)
y_head = rf.predict(x_)

plt.scatter(x,y,color = "red")
plt.plot(x_,y_head, color = "green")
#tahmin etmek istediğim değer, tahmin sonucu
plt.xlabel("tribün level")
plt.ylabel("ücret")
plt.show()



#%%

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(x,y)

y_head = rf.predict(x)


#%%

#R Square

from sklearn.metrics import r2_score

print("r 2", r2_score(y,y_head))












