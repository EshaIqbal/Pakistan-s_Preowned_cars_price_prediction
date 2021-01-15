# -*- coding: utf-8 -*-
"""Honda"""

#uploading file
from google.colab import files 
  
  
uploaded = files.upload()

#importing libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#importing set
import io 
        
data = pd.read_csv(io.BytesIO(uploaded['Honda.csv'])) 
data.head()

#PreProcessing
#dropping features
data.drop(['Steering_Switches','Rear_seat_Entertainment','Rear_Camera','Rear_Speakers','Rear_AC_Vents','Front_Camera','Heated_Seats','Climate_Control','Toolbox'], axis = 1, inplace = True) 
data.head()

#finding nan values
data.isna().sum()

#dropping nan values
dataset=data.dropna()

#get_dumies
# generate binary values using get_dummies
Engine_Type = pd.get_dummies( dataset['Engine_Type'],columns=['Engine_Type'], prefix=['Engine_Type_is'] )
Engine_Type.head()

#now joing colomns using join()
dataset = dataset.join(Engine_Type)
#dropping EngineType
dataset.drop(['Engine_Type'], axis=1, inplace=True)
dataset

# dummies of Body_Type
Body_Type = pd.get_dummies( dataset['Body_Type'],columns=['Body_Type'], prefix=['Body_Type_is'] )
Body_Type.head()

#dropping Body_Type
dataset.drop(['Body_Type'], axis=1, inplace=True)
dataset
#now joing colomns using join()
dataset = dataset.join(Body_Type)

#dummies of color
Color= pd.get_dummies(dataset.Color,columns=['Color'], prefix=['Color'] )
Color.head()

#now joing colomns using join()
dataset = dataset.join(Color)
#dropping color
dataset.drop(['Color'], axis=1, inplace=True)
dataset

# creating instance of labelencoder
labelencoder = LabelEncoder()
# Assigning numerical values and storing in another column
dataset['Model_Name'] = labelencoder.fit_transform(dataset['Name'])
dataset.head()

# generate binary values using get_dummies
Name = pd.get_dummies(dataset.Name, columns=["Name"], prefix=["Model"] )
Name

#now joing colomns using join()
Honda_cars = dataset.join(Name)
#dropping Names
Honda_cars.drop(['Name'], axis=1, inplace=True)
Honda_cars

Honda_cars.fillna(0)

Honda_cars['Anti_lock_Braking_System']=Honda_cars['Anti_lock_Braking_System'].fillna(0)

Honda_cars.shape

Honda_cars.describe()

#Machine Learning
y =Honda_cars['Price']
Honda_cars.drop(['Price'], axis=1, inplace=True)
X = Honda_cars


#trainging model testing data is less and learning is more
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) 
#randome_state=None it return different result after each excution
#randome_state=interger it return same result after each excution

#Training the Algorithm
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
coeff_df

Honda_cars = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
Honda_cars

from sklearn import metrics
import sklearn.metrics as sm
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_pred))) 
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_pred))) 
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_pred))) 
print("Explain variance score =", (sm.explained_variance_score(y_test, y_pred)))
print("R2 score =", (sm.r2_score(y_test, y_pred)))

#Deep learning
import numpy as np
import pandas as pd



import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

print(tf.__version__)

model = keras.models.Sequential()

model.add(keras.layers.Dense(237, activation='relu', input_shape=(237,)))
model.add(keras.layers.Dense(237, activation='relu'))
model.add(keras.layers.Dense(1))
model.summary() #param shows connection of neuron with neorons of second layer

keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='mean_squared_error', optimizer='RMSprop', metrics=['mean_absolute_percentage_error'])
model.fit(X, y, epochs=1000, batch_size=10,callbacks=[keras.callbacks.EarlyStopping(patience=3)])

#Prediction
pred=model.predict(X_test)

pred=pd.DataFrame(pred)
actual=pd.DataFrame(y_test)

actual=pd.DataFrame(y_test)
actual.head()

pred.columns=['Pred']
pred.head()

#Reindexing
a=[]
for i in range (1887):
  a.append(i)
actual.index=a
Hondacars=pd.concat([pred, actual], axis=1)
Hondacars.head(30)

from sklearn import metrics
import sklearn.metrics as sm
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, pred))) 
print("Mean squared error =", round(sm.mean_squared_error(y_test, pred))) 
print("Median absolute error =", round(sm.median_absolute_error(y_test, pred))) 
print("Explain variance score =", (sm.explained_variance_score(y_test, pred)))
print("R2 score =", (sm.r2_score(y_test, pred)))

#Graph

import matplotlib.pyplot as plt
plt.figure(figsize=(100,200))
fig, ax = plt.subplots()
ax.plot(Hondacars.index,Hondacars.Pred,color='blue',marker='o')
ax.plot(Hondacars.index,Hondacars.Price, color='orange',marker='*')
