# -*- coding: utf-8 -*-
"""Mercedes.ipynb"""

import keras
import numpy as np
import pandas as pd
import io

from google.colab import files 
  
  
uploaded = files.upload()

mercedes = pd.read_csv(io.BytesIO(uploaded['Mercedes.csv']))

mercedes

namedum = pd.get_dummies(mercedes.Name)
mercedes = mercedes.join(namedum)
colourdum = pd.get_dummies(mercedes.Color,prefix="is")
mercedes=mercedes.join(colourdum)
enginedum = pd.get_dummies(mercedes['Engine Type'])
mercedes=mercedes.join(enginedum)
Bodydum = pd.get_dummies(mercedes['Body Type'])
mercedes=mercedes.join(Bodydum)
mercedes.drop(['Name', 'Color', 'Engine Type', 'Body Type'], axis=1, inplace=True)

mercedes['Anti lock Braking System']=mercedes['Anti lock Braking System'].fillna(0)

mercedes.shape

Y = mercedes.Price
mercedes.drop(['Price'], axis=1, inplace=True)
X=mercedes

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25,random_state=0)

model = keras.models.Sequential()

model.add(keras.layers.Dense(128, activation='relu', input_shape=(81,)))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='linear'))

keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_percentage_error'])

model.fit(X, Y, epochs=10000,batch_size=10, callbacks=[keras.callbacks.EarlyStopping(patience=3)])

pred=model.predict(X_test, batch_size=1)

actual=pd.DataFrame(Y_test) 
pred=pd.DataFrame(pred)

actual.shape

actual

pred.columns=['Pred']

a=[]
for i in range (123):
  a.append(i)
actual.index=a

actual

mercedes=pd.concat([pred, actual], axis=1)

mercedes

from sklearn import metrics
import sklearn.metrics as sm
print("Mean absolute error =", round(sm.mean_absolute_error(mercedes.Pred, mercedes.Price))) 
print("Mean squared error =", round(sm.mean_squared_error(mercedes.Pred, mercedes.Price)))
print("Median absolute error =", round(sm.median_absolute_error(mercedes.Pred, mercedes.Price))) 
print("Explain variance score =", (sm.explained_variance_score(mercedes.Pred, mercedes.Price)))
print("R2 score =", (sm.r2_score(mercedes.Pred, mercedes.Price)))

