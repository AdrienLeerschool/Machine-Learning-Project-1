"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classical algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.tree import DecisionTreeClassifier

from data import make_data1, make_data2
from plot import plot_boundary
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# (Question 1)

# Put your funtions here
# ...


if __name__ == "__main__":
    
    depths = [1, 2, 4, 8, 0]
    [X_train, y_train, X_test, y_test] = make_data1(10000,250,0,None)
    print("Data set 1 :")
    
    for i in depths:
        if i == 0 :
            clf = DecisionTreeClassifier(max_depth=None)
        else:
            clf = DecisionTreeClassifier(max_depth=i)
        clf = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        if i == 0 :
            print("Accuracy for depth = None : "+ str(accuracy_score(y_test, y_pred)))
            plot_boundary(fname="data1_depthNone",fitted_estimator=clf,
                          X=X_test,y=y_pred,title="First data set with depth = None")
        else:
            print("Accuracy for depth = "+str(i)+" : "+ str(accuracy_score(y_test, y_pred)))
            plot_boundary(fname="data1_depth"+str(i),fitted_estimator=clf,
                          X=X_test,y=y_pred,title="First data set with depth = "+str(i))
        
# =============================================================================
#         X1 = X_test[y_test==0]
#         X2 = X_test[y_test==1]
#         plt.scatter(X1[:,0], X1[:,1], color="DodgerBlue")
#         plt.scatter(X2[:,0], X2[:,1], color="darkorange")
#         plt.plot([0], [0], marker="x", markersize=10)
#         plt.grid(True)
#         plt.xlabel("X_0")
#         if i == 0 :
#             plt.ylabel("X_1 depth = None")
#             print("Accuracy for depth = None : "+ str(accuracy_score(y_test, y_pred)))
#         else:
#             plt.ylabel("X_1 depth = "+str(i))
#             print("Accuracy for depth = "+str(i)+" : "+ str(accuracy_score(y_test, y_pred)))
#         plt.show()
# =============================================================================
        
    
    
    [X_train, y_train, X_test, y_test] = make_data2(10000,250,0,None)
    print("\nData set 2 :")
    for i in depths:
        if i == 0 :
            clf = DecisionTreeClassifier(max_depth=None)
        else:
            clf = DecisionTreeClassifier(max_depth=i)
        clf = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        if i == 0 :
            print("Accuracy for depth = None : "+ str(accuracy_score(y_test, y_pred)))
            plot_boundary(fname="data2_depthNone",fitted_estimator=clf,
                          X=X_test,y=y_pred,title="Second data set with depth = None")
        else:
            print("Accuracy for depth = "+str(i)+" : "+ str(accuracy_score(y_test, y_pred)))
            plot_boundary(fname="data2_depth"+str(i),fitted_estimator=clf,
                          X=X_test,y=y_pred,title="Second data set with depth = "+str(i))
    
# =============================================================================
#         X1 = X_test[y_test==0]
#         X2 = X_test[y_test==1]
#         plt.scatter(X1[:,0], X1[:,1], color="DodgerBlue")
#         plt.scatter(X2[:,0], X2[:,1], color="darkorange")
#         plt.plot([0], [0], marker="x", markersize=10)
#         plt.grid(True)
#         plt.xlabel("X_0")
#         if i == 0 :
#             plt.ylabel("X_1 depth = None")
#             print("Accuracy for depth = None : "+ str(accuracy_score(y_test, y_pred)))
#         else:
#             plt.ylabel("X_1 depth = "+str(i))
#             print("Accuracy for depth = "+str(i)+" : "+ str(accuracy_score(y_test, y_pred)))
#         plt.show()
# =============================================================================
