# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 11:34:37 2018

@author: cookchr2
"""


#importing the necessary packages
import pandas as pd
import os
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix
import time

#get the start time
t0 = time.time()


#fix random seed for reproducibility
seed = 21

#This let's numpy use our random seed
np.random.seed(seed)

#This is the directory 
os.chdir('R:/JoePriceResearch/record_linking/projects/deep_learning/census')

df = pd.read_stata('virpilot_machine_take2dl.dta')

df['sort'] = pd.Series(np.random.uniform(size=df.shape[0]))

df['m_dist'] = df['distance'] == df['distance'].min()
df['m_byr'] = df['poi_birthyear'] == df['poi_birthyear'].min()

#df = df.drop(['same_fbpl','same_mbpl'],axis=1)

Y = pd.DataFrame()
Y['true'] = df['true']
Y['false'] = 1 - df['true']
Y['sort'] = df['sort']


X, Xtest, Y, Ytest = train_test_split(df,Y,test_size = 0.2, random_state=2222)

#Surprisingly another hyperparameter! What percentage of the false matches do we keep?
Y = Y[(Y.sort >= 0.5) | (Y.true == 1)]
X = X[(X.sort >= 0.5) | (X.true == 1)]


X = X.drop(['true','sort'], axis=1)
Y = Y.drop(['sort'], axis=1)
Ytest = Ytest.drop(['sort'], axis=1)
Xtest = Xtest.drop(['sort'], axis=1)

X['poi_birthyear'] = (X['poi_birthyear'] - df['poi_birthyear'].mean()) /  df['poi_birthyear'].std()
Xtest['poi_birthyear'] = (Xtest['poi_birthyear'] - df['poi_birthyear'].mean()) /  df['poi_birthyear'].std()


Ytestt = Ytest[(Ytest.true == 1)]
Xtestt = Xtest[(Xtest.true == 1)]
Ytestf = Ytest[(Ytest.true == 0)]
Xtestf = Xtest[(Xtest.true == 0)]

Xtestt = Xtestt.drop(['true'], axis=1)
Xtestf = Xtestf.drop(['true'], axis=1)




# create a model
model = Sequential([Dense(50, input_dim=48, activation="relu", kernel_initializer='he_normal'),
                    BatchNormalization(),
                    Dropout(0.5),
                    Dense(25, activation="relu", kernel_initializer='he_normal'),
                    BatchNormalization(),
                    Dropout(0.5),
                    Dense(12, activation="relu", kernel_initializer='he_normal'),
                    BatchNormalization(),
                    Dropout(0.5),
                    Dense(6, activation="relu", kernel_initializer='he_normal'),
                    BatchNormalization(),
                    Dropout(0.5),
                    Dense(3, activation="relu", kernel_initializer='he_normal'),
                    BatchNormalization(),
                    Dropout(0.3),
                    Dense(2, activation="softmax", kernel_initializer='he_normal')
                    
        ])


#Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#fit 

model.fit(X.values, Y.values, epochs=4, batch_size=16,validation_split=0.1)


#save the recently trained model

model.save('R:/JoePriceResearch/record_linking/projects/deep_learning/census/model21.h5')
#evalute

#Get a score for true
scores = model.evaluate(Xtestt.values, Ytestt.values)
print('True %s: %.f%%' % (model.metrics_names[1], scores[1]*1000))


scores = model.evaluate(Xtestf.values, Ytestf.values)
print('False %s: %.f%%' % (model.metrics_names[1], scores[1]*1000))

'''
test_predictions = np.argmax(model.predict(Xtest),1)
y_test_sparse = np.argmax(Ytest, 1)
print(confusion_matrix(y_test_sparse, test_predictions))
'''
t1 = time.time()

total = t1-t0
print(total/60)
