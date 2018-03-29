# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 11:34:37 2018

@author: cookchr2
"""


#importing the necessary packages
import pandas as pd
import os
import numpy
import time
from sklearn import linear_model
from sklearn import svm
from sklearn import tree
from sklearn import ensemble
from sklearn import model_selection
from sklearn import metrics
import pickle


#fix random seed for reproducibility
seed = 21

#This let's numpy use our random seed
numpy.random.seed(seed)

#This is the directory 
os.chdir('R:/JoePriceResearch/record_linking/projects/deep_learning/census')

X = pd.read_stata('virpilot_machine_take2dl.dta')

X['sort'] = pd.Series(numpy.random.uniform(0.0,1.0,X['true'].count()))
X = X[(X.sort > 0.6) | (X.true == 1)]

X = X.drop(['same_fbpl','same_mbpl','sort'],axis=1)

Y = X['true']

X, Xtest, Y, Ytest = model_selection.train_test_split(X, Y, test_size=0.20, random_state=21)

Ytestf = Ytest[(Ytest == 0)]
Xtestf = Xtest[(Xtest.true == 0)]
Ytest = Ytest[(Ytest == 1)]
Xtest = Xtest[(Xtest.true == 1)]



X = X.drop(['true'], axis=1)
Xtest = Xtest.drop(['true'], axis=1)
Xtestf = Xtestf.drop(['true'], axis=1)


# create some models
rf_gs = ensemble.RandomForestClassifier(n_estimators = 1000, max_features = 0.8, max_depth = 15, n_jobs = -1)
#rf_gs = model_selection.GridSearchCV(rf, {'n_estimators' : [1000], 'max_features':[None], 'max_depth':[20], 'n_jobs':[-1] })



#fit  models
#get the start time
t0 = time.time()
print('starting')
rf_gs.fit(X.values, Y.values)
t1 = time.time()

in_rf_acc = rf_gs.score(X.values, Y.values)
print('In sample predictions: ' + str(in_rf_acc))


#save the recently trained model

#model.save('R:/JoePriceResearch/record_linking/projects/deep_learning/census/model19.h5')
#evalute


rf_acct = rf_gs.score(Xtest.values, Ytest.values)
rf_accf = rf_gs.score(Xtestf.values, Ytestf.values)

print('True: ' + str(rf_acct))
print('False: ' + str(rf_accf))


total = t1-t0
print('RF')
print(total/60)
#print("Best Params: {}".format(rf_gs.best_params_))


#Save models
pickle.dump(rf_gs, open('R:/JoePriceResearch/record_linking/projects/deep_learning/census/rf5.sav', 'wb'))


