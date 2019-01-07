# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 23:51:28 2018

@author: Sena
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv")

data.info()

data.drop(["Unnamed: 32","id"], axis = 1, inplace = True)
# axis = 1 -> tüm sütunu drop et
# inplace = True -> yerine yerleştir
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]


y = data.diagnosis.values
x_data = data.drop(["diagnosis"], axis = 1)

#%% normalization
# çok büyük değerler diğer feature ları baskılar bu da modeli bozar
# modeli bozan feeature ları normalize ederiz

x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data)).values

# (x - minx) / (max(x) - min(x) )


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)
#datamızı ayırıyoruz
#x train ve y trainle datayı ayırdık, 
#x test ve y test le test ettik
# y = label
#x = feature
#test size -> % 20 si test %80 train
#random state -> random şekilde, 42 değerine göre böler
#42 önemli daha sonra da 42 modeline göre bölünmüşünü kullancaz

x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T


#%%% parameter initialize ve sigmoid function

#dimension = pixel (4096) yani feature lar
def initialize_weight_bias(dimension):
    




















