"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classical algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.neighbors import KNeighborsClassifier

from data import make_data1, make_data2
from plot import plot_boundary
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# (Question 2)

# Put your funtions here
# ...

def make_model(size_ts = 10000, size_ls = 250, data_set = 1, graph = False, n_neigh = 1, cv = False):
    
    if data_set == 1 :
        [X_train, y_train, X_test, y_test] = make_data1(size_ts,size_ls,0,None)
    else:
        [X_train, y_train, X_test, y_test] = make_data2(size_ts,size_ls,0,None)

    clf = KNeighborsClassifier(n_neighbors=n_neigh)
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
                            
    
    if graph :
        plot_boundary(fname="figures/data_" + str(data_set) + "_neighbors"+str(n), fitted_estimator=clf, 
                      X=X_test, y=y_pred, title="data set " + str(data_set) + " with neighbors = "+str(n))
        
    return accuracy_score(y_test, y_pred)


def cross_validation_function(k, X_train, y_train, ls_size):
    
    five_folds = [0] * 200
    subsets_X = np.split(X_train, k)
    subsets_Y = np.split(y_train, k)
    
    
    for i in list(range(0, k)):
        accur = []
        
        cv_test = subsets_X[i]
        cv_test_y = subsets_Y[i]
        
        arr = np.delete(subsets_X, i, 0)
        cv_train = np.concatenate(arr)
        arr = np.delete(subsets_Y, i, 0)
        cv_train_y = np.concatenate(arr)
        
        n_neighs = list(range(1, 200))
        
        for n_neigh in n_neighs:
        
            clf = KNeighborsClassifier(n_neighbors=n_neigh)
            clf = clf.fit(cv_train, cv_train_y)
            y_pred = clf.predict(cv_test)
            accur.append(accuracy_score(cv_test_y, y_pred))
            five_folds[n_neigh] = five_folds[n_neigh] + accuracy_score(cv_test_y, y_pred)
        

        best = five_folds.index(max(five_folds))
        acc_rate = max(five_folds) / k
#        print('Best neighbor is ' + str(best) + ' with an accuracy of ' + str(acc_rate) + ' ! ')
#        plt.plot(n_neighs, accur)
#        plt.xlabel('Neighbors')
#        plt.ylabel('Accuracy')
    
    return best, acc_rate
    
    

if __name__ == "__main__":

    n_neighbors = [1, 5, 10, 75, 100, 150]
    score = []
    best_arr = []
    acc_rate_arr = []
    iteration = 100

    
    # First data set
    print("First data set :\n")
    
    #Q1       
    for n in n_neighbors :
        score = make_model(n_neigh = n, data_set = 1, graph = True)
        print("Accuracy for n_neighbors " + str(n) + " : " + str(score))
    
    
    #Q3 
    LS_size = [50, 200, 250, 500]
    
    for size in LS_size:
        res = []
        x = list(range(1, size))
        [X_train, y_train, X_test, y_test] = make_data1(500, size, 0, None)
            
        for n in x :
            clf = KNeighborsClassifier(n_neighbors=n)
            clf = clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            res.append(accuracy_score(y_test, y_pred))
        
        plt.figure()
        plt.plot(x, res)
        plt.xlabel('Neighbors')
        plt.ylabel('Accuracy')
        plt.savefig('data_1_'+ str(size))
        

      
    # Second data set
    print("\n\nSecond data set :\n")
   
    #Q1
    for n in n_neighbors :
        score = make_model(n_neigh = n, data_set = 2, graph = True)
        print("Accuracy for n_neighbors " + str(n) + " : " + str(score))

    
    #Q2
    #Cross validation 
    for i in list(range(1, iteration)):
        [X_train, y_train, X_test, y_test] = make_data2(10000, 250, 0, False ,None)
        best, acc_rate = cross_validation_function(5, X_train, y_train, 200)
        best_arr.append(best)
        acc_rate_arr.append(acc_rate)
    
    counts = np.bincount(best_arr)
    print(np.argmax(counts))
    mean = np.mean(best_arr)
    
    #Q3          
    LS_size = [50, 200, 250, 500]
    
    for size in LS_size:
        res = []
        x = list(range(1, size))
        [X_train, y_train, X_test, y_test] = make_data2(500, size, 0, None)
            
        for n in x :
            clf = KNeighborsClassifier(n_neighbors=n)
            clf = clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            res.append(accuracy_score(y_test, y_pred))
        
        plt.figure()
        plt.plot(x, res)
        plt.xlabel('Neighbors')
        plt.ylabel('Accuracy')
        plt.savefig('data_2_'+ str(size))
