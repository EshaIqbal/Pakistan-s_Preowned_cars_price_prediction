# -*- coding: utf-8 -*-
"""Hyundai.ipynb"""

import keras
import numpy as np
import pandas as pd
import io

from google.colab import files 
  
  
uploaded = files.upload()

hyundai = pd.read_csv(io.BytesIO(uploaded['Hyundai.csv']))

hyundai

namedum = pd.get_dummies(hyundai.Name)
hyundai = hyundai.join(namedum)
colourdum = pd.get_dummies(hyundai.Color,prefix="is")
hyundai=hyundai.join(colourdum)
enginedum = pd.get_dummies(hyundai['Engine Type'])
hyundai=hyundai.join(enginedum)
Bodydum = pd.get_dummies(hyundai['Body Type'])
hyundai=hyundai.join(Bodydum)
hyundai.drop(['Name', 'Color', 'Engine Type', 'Body Type'], axis=1, inplace=True)

hyundai['Anti lock Braking System']=hyundai['Anti lock Braking System'].fillna(0)

hyundai.shape

Y = hyundai.Price
hyundai.drop(['Price'], axis=1, inplace=True)
X=hyundai

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25,random_state=40)

model = keras.models.Sequential()

model.add(keras.layers.Dense(128, activation='relu', input_shape=(57,)))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(8, activation='relu'))
model.add(keras.layers.Dense(1, activation='linear'))

keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse','mae','acc'])

model.fit(X, Y, epochs=6000,batch_size=40, callbacks=[keras.callbacks.EarlyStopping(patience=3)])

pred=model.predict(X_test, batch_size=1)

actual=pd.DataFrame(Y_test) 
pred=pd.DataFrame(pred)

actual.shape

pred.columns=['Pred']

a=[]
for i in range (116):
  a.append(i)
actual.index=a

hyundai=pd.concat([pred, actual], axis=1)

hyundai

from sklearn import metrics
import sklearn.metrics as sm
print("Mean absolute error =", round(sm.mean_absolute_error(hyundai.Pred, hyundai.Price))) 
print("Mean squared error =", round(sm.mean_squared_error(hyundai.Pred, hyundai.Price)))
print("Median absolute error =", round(sm.median_absolute_error(hyundai.Pred, hyundai.Price))) 
print("Explain variance score =", (sm.explained_variance_score(hyundai.Pred, hyundai.Price)))
print("R2 score =", (sm.r2_score(hyundai.Pred, hyundai.Price)))

import matplotlib.pyplot as plt
plt.figure(figsize=(100,200))
fig, ax = plt.subplots()
ax.plot(hyundai.index,hyundai['Pred'],color='blue',marker='o')
ax.plot(hyundai.index,hyundai['Price'], color='orange',marker='*')
ax.legend(labels=['Predicted', 'Actual'])

pred=0
act=0
for i in range (116):
  pred=pred+hyundai.Pred[i]
  act=act+hyundai.Price[i]
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
langs = ['Actual','Predicted']
students = [act,pred]
ax.bar(langs[0],students[0],color='black')
ax.bar(langs[1],students[1],color='blue')
plt.show()

import numpy as np
import random
a=random.randrange(116)
b=random.randrange(116)
c=random.randrange(116)
d=random.randrange(116)
e=random.randrange(116)
f=random.randrange(116)
g=random.randrange(116)
h=random.randrange(116)
i=random.randrange(116)
j=random.randrange(116)
data= [[hyundai.Pred[a],hyundai.Pred[b],hyundai.Pred[c],hyundai.Pred[d],hyundai.Pred[e],hyundai.Pred[f],hyundai.Pred[g],hyundai.Pred[h],hyundai.Pred[i],hyundai.Pred[j]],
     [hyundai.Price[a],hyundai.Price[b],hyundai.Price[c],hyundai.Price[d],hyundai.Price[e],hyundai.Price[f],hyundai.Price[g],hyundai.Price[h],hyundai.Price[i],hyundai.Price[j]]]

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
X=np.arange(10)
ax.bar(X + 0.00, data[0], color = 'b', width = 0.25,)
ax.bar(X + 0.25, data[1], color = 'black', width = 0.25,)
ax.legend(labels=['Predicted', 'Actual'])

