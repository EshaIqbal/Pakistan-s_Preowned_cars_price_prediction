# -*- coding: utf-8 -*-
"""Mitsubshi.ipynb"""

import keras
import numpy as np
import pandas as pd
import io
import matplotlib.pyplot as plt

from google.colab import files 
  
  
uploaded = files.upload()

mitsubshi= pd.read_csv(io.BytesIO(uploaded['Mitsubishi.csv'])) 
mitsubshi

namedum = pd.get_dummies(mitsubshi.Name)
mitsubshi = mitsubshi.join(namedum)
colourdum = pd.get_dummies(mitsubshi.Color,prefix="is")
mitsubshi=mitsubshi.join(colourdum)
enginedum = pd.get_dummies(mitsubshi['Engine Type'])
mitsubshi=mitsubshi.join(enginedum)
Bodydum = pd.get_dummies(mitsubshi['Body Type'])
mitsubshi=mitsubshi.join(Bodydum)
mitsubshi.drop(['Name', 'Color', 'Engine Type', 'Body Type'], axis=1, inplace=True)

mitsubshi['Anti lock Braking System']=mitsubshi['Anti lock Braking System'].fillna(0)

mitsubshi= mitsubshi.fillna(0)

mitsubshi.shape

Y = mitsubshi.Price

mitsubshi.drop(['Price'], axis=1, inplace=True)
X=mitsubshi

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=0)

model = keras.models.Sequential()

model.add(keras.layers.Dense(128, activation='relu', input_shape=(118,)))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='linear'))

keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_percentage_error'])

keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse','mae','acc'])

model.fit(X, Y, epochs=10000,batch_size=85, callbacks=[keras.callbacks.EarlyStopping(patience=3)])

pred=model.predict(X_test, batch_size=1)

actual=pd.DataFrame(Y_test) 
pred=pd.DataFrame(pred)

pred.columns=['Pred']

actual.shape

a=[]
for i in range (152):
  a.append(i)
actual.index=a

mitsubshi=pd.concat([pred, actual], axis=1)

mitsubshi

from sklearn import metrics
import sklearn.metrics as sm
print("Mean absolute error =", round(sm.mean_absolute_error(mitsubshi.Pred, mitsubshi.Price))) 
print("Mean squared error =", round(sm.mean_squared_error(mitsubshi.Pred, mitsubshi.Price)))
print("Median absolute error =", round(sm.median_absolute_error(mitsubshi.Pred, mitsubshi.Price))) 
print("Explain variance score =", (sm.explained_variance_score(mitsubshi.Pred, mitsubshi.Price)))
print("R2 score =", (sm.r2_score(mitsubshi.Pred, mitsubshi.Price)))

