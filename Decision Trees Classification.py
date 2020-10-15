# -*- coding: utf-8 -*-
"""
Created on Tue May 26 07:23:06 2020

@author: Mutunga
"""

#DECISION TREES CLASSFIER

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('Credit_Cards_Default.csv')
dataset.info()
dataset.head()

#Then X = dataset.drop('Purchased', axis=1) and y = dataset['Purchased']
#Alternatively

X=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values
print(X)
print(y)


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


#Training the Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy',max_depth=3, min_samples_leaf=10, random_state=0) #or gini
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
Accuracy = np.round(accuracy_score(y_test, y_pred),4)
print('R Squared Score:', Accuracy*100, '%')

#Classification report
from sklearn.metrics import classification_report
CR=classification_report(y_test, y_pred)
print(CR)

#Visualizing the Descision Tree
from sklearn import tree
import graphviz
dataset_feature_names = list(X_train)
dot_data = tree.export_graphviz(classifier, feature_names=None)
graphviz.Source(dot_data)

#Visualizing a Decision Tree
from IPython.display import Image
import pydotplus
from sklearn.tree import export_graphviz
import collections

dot_data = export_graphviz(classifier,
                                feature_names= None,
                                out_file=None,
                                filled=True,
                                rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)

colors = ('turquoise', 'orange')
edges = collections.defaultdict(list)

for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()    
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])

graph.write_png('tree.png')