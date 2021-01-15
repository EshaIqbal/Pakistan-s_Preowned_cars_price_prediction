# -*- coding: utf-8 -*-
#Nissan.ipynb


import keras
import numpy as np
import pandas as pd
import io
import matplotlib.pyplot as plt

from google.colab import files 
  
  
uploaded = files.upload()

nissan= pd.read_csv(io.BytesIO(uploaded['Nissan.csv'])) 
nissan

namedum = pd.get_dummies(nissan.Name)
nissan = nissan.join(namedum)
colourdum = pd.get_dummies(nissan.Color,prefix="is")
nissan=nissan.join(colourdum)
enginedum = pd.get_dummies(nissan['Engine Type'])
nissan=nissan.join(enginedum)
Bodydum = pd.get_dummies(nissan['Body Type'])
nissan=nissan.join(Bodydum)
nissan.drop(['Name', 'Color', 'Engine Type', 'Body Type'], axis=1, inplace=True)

nissan['Anti lock Braking System']=nissan['Anti lock Braking System'].fillna(0)

nissan.shape

Y = nissan.Price

nissan.drop(['Price'], axis=1, inplace=True)
X=nissan

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=0)

model = keras.models.Sequential()

model.add(keras.layers.Dense(128, activation='relu', input_shape=(149,)))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(8, activation='relu'))
model.add(keras.layers.Dense(1, activation='linear'))

keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_percentage_error'])

keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse','mae','acc'])

model.fit(X, Y, epochs=4500,batch_size=35, callbacks=[keras.callbacks.EarlyStopping(patience=3)])

pred=model.predict(X_test, batch_size=1)

actual=pd.DataFrame(Y_test) 
pred=pd.DataFrame(pred)

pred.columns=['Pred']

actual.shape

a=[]
for i in range (212):
  a.append(i)
actual.index=a

nissan=pd.concat([pred, actual], axis=1)

nissan

from sklearn import metrics
import sklearn.metrics as sm
print("Mean absolute error =", round(sm.mean_absolute_error(nissan.Pred, nissan.Price))) 
print("Mean squared error =", round(sm.mean_squared_error(nissan.Pred, nissan.Price)))
print("Median absolute error =", round(sm.median_absolute_error(nissan.Pred, nissan.Price))) 
print("Explain variance score =", (sm.explained_variance_score(nissan.Pred, nissan.Price)))
print("R2 score =", (sm.r2_score(nissan.Pred, nissan.Price)))

plt.figure(figsize=(100,200))
fig, ax = plt.subplots()
ax.plot(nissan.index,nissan['Pred'],color='blue',marker='o')
ax.plot(nissan.index,nissan['Price'], color='orange',marker='*')

pred=0
act=0
for i in range (212):
  pred=pred+nissan.Pred[i]
  act=act+nissan.Price[i]

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
langs = ['Actual','Predicted']
students = [act,pred]
ax.bar(langs[0],students[0],color='black')
ax.bar(langs[1],students[1],color='blue')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import random
a=random.randrange(212)
b=random.randrange(212)
c=random.randrange(212)
d=random.randrange(212)
e=random.randrange(212)
f=random.randrange(212)
g=random.randrange(212)
h=random.randrange(212)
i=random.randrange(212)
j=random.randrange(212)
data= [[nissan.Pred[a],nissan.Pred[b],nissan.Pred[c],nissan.Pred[d],nissan.Pred[e],nissan.Pred[f],nissan.Pred[g],nissan.Pred[h],nissan.Pred[i],nissan.Pred[j]],
     [nissan.Price[a],nissan.Price[b],nissan.Price[c],nissan.Price[d],nissan.Price[e],nissan.Price[f],nissan.Price[g],nissan.Price[h],nissan.Price[i],nissan.Price[j]]]

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
X=np.arange(10)
ax.bar(X + 0.00, data[0], color = 'b', width = 0.25,)
ax.bar(X + 0.25, data[1], color = 'black', width = 0.25,)
ax.legend(labels=['Predicted', 'Actual'])
