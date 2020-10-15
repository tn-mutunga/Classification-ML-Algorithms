# -*- coding: utf-8 -*-
"""
Created on Mon May 25 07:45:48 2020

@author: Mutunga
"""
#SUPPORT VECTOR MACHINES

#For data that is not linearly seperable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('Credit_Cards_Default.csv')
dataset.info()

X=dataset.iloc[:,2:-1].values
y=dataset.iloc[:,-1].values


#Splitting dataset into Training and Test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.25, random_state=0)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

#Feature scaling print

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train)
print(X_test)


#Training the SVM
#Check API from scikit website...choose SVCLinear or SVC(with linear options)
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0) #Classic SVM
classifier.fit(X_train, y_train)
 
#Predicting the test results
y_pred = classifier.predict(X_test)
np.set_printoptions(precision=2)
Test_vs_Predicted=np.concatenate((y_test.reshape(len(y_test),1), y_pred.reshape(len(y_pred),1)),1)
print(Test_vs_Predicted)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score #or from sklearn.metrics import confusion_matrix, accuracy_score
Accuracy = np.round(accuracy_score(y_test, y_pred),2)
print('R Squared Score:', Accuracy*100, '%')

#Classification report
from sklearn.metrics import classification_report
CR=classification_report(y_test, y_pred)
print(CR)


