# -*- coding: utf-8 -*-
"""suzuki"""

from google.colab import files 
  
  
uploaded = files.upload()

#importing libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import io 
        
data = pd.read_csv(io.BytesIO(uploaded['Suzuki.csv'])) 
data

#Pre Processing
data.isna().sum()

data['Anti_lock_Braking_System']=data['Anti_lock_Braking_System'].fillna(0)

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
Suzuki_cars = dataset.join(Name)
#dropping Names
Suzuki_cars.drop(['Name'], axis=1, inplace=True)
Suzuki_cars

Suzuki_cars.fillna(0)

Suzuki_cars['Anti_lock_Braking_System']=Suzuki_cars['Anti_lock_Braking_System'].fillna(0)

Suzuki_cars.shape

Suzuki_cars.describe()

#Machine Learning
y =Suzuki_cars['Price']
Suzuki_cars.drop(['Price'], axis=1, inplace=True)
X = Suzuki_cars

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

Suzuki_cars = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
Suzuki_cars

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

model.add(keras.layers.Dense(225, activation='relu', input_shape=(225,)))
model.add(keras.layers.Dense(225, activation='relu'))
model.add(keras.layers.Dense(1))
model.summary()

keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='mean_squared_error', optimizer='RMSprop', metrics=['mean_absolute_percentage_error'])
model.fit(X, y, epochs=1000, batch_size=10 ,callbacks=[keras.callbacks.EarlyStopping(patience=3)])

pred=model.predict(X_test)

pred=pd.DataFrame(pred)
actual=pd.DataFrame(y_test)

actual=pd.DataFrame(y_test)
actual

pred.columns=['Pred']
pred

#Reindexing
a=[]
for i in range (3217):
  a.append(i)
actual.index=a
Suzuki_cars=pd.concat([pred, actual], axis=1)
Suzuki_cars

from sklearn import metrics
import sklearn.metrics as sm
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, pred))) 
print("Mean squared error =", round(sm.mean_squared_error(y_test, pred))) 
print("Median absolute error =", round(sm.median_absolute_error(y_test, pred))) 
print("Explain variance score =", (sm.explained_variance_score(y_test, pred)))
print("R2 score =", (sm.r2_score(y_test, pred)))

