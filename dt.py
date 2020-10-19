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

from sklearn import tree
import graphviz 


def make_model(size_ts = 10000, size_ls = 250, data_set = 1, graph = False, depth = None, random_state = 0):
    
    if data_set == 1 :
        [X_train, y_train, X_test, y_test] = make_data1(size_ts,size_ls,0,False,random_state=random_state)
    else:
        [X_train, y_train, X_test, y_test] = make_data2(size_ts,size_ls,0,False,random_state=random_state)

    clf = DecisionTreeClassifier(max_depth=depth)
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    if graph:
        if depth == None :
            plot_boundary(fname="figures/data"+str(data_set)+"_depthNone",fitted_estimator=clf,
                          X=X_test,y=y_pred,title="data set "+str(data_set)+" with depth = None")
            
            tree.plot_tree(clf)
            dot_data = tree.export_graphviz(clf, out_file=None)
            graph = graphviz.Source(dot_data)
            graph.render("figures/tree_data"+str(data_set)+"_depthNone")
        
        else:
            plot_boundary(fname="figures/data"+str(data_set)+"_depth"+str(depth),fitted_estimator=clf,
                          X=X_test,y=y_pred,title="data set "+str(data_set)+" with depth = "+str(depth))
                   
    return accuracy_score(y_test, y_pred)



if __name__ == "__main__":
    
    depths = [1, 2, 4, 8, 0]
    scores = []
    nb_iter = 4
    
    # First data set
    print("First data set :\n")
    
    for i in depths :
        scores.clear()
        for j in range(nb_iter) :
            if i == 0 :
                scores.append(make_model(depth=None,graph=True))
            else:
                scores.append(make_model(depth=i,graph=True))
                
        if i == 0 :
            scores.append(make_model(depth=None, graph=True))
            print("Average accuracy for unconstrained depth : "+ str(np.mean(scores)) +
                  ",\nstandard deviation : "+str(np.std(scores)))
        else:
            scores.append(make_model(depth=i,graph=True))
            print("Average accuracy for max_depth = "+str(i)+" : "+ str(np.mean(scores))+
                  ",\nstandard deviation : "+str(np.std(scores)))
      
    # Second data set
    print("\n\nSecond data set :\n")
        
    for i in depths :
        scores.clear()
        for j in range(nb_iter) :
            if i == 0 :
                scores.append(make_model(depth=None, data_set=2,graph=True))
            else:
                scores.append(make_model(depth=i, data_set=2,graph=True))
                
        if i == 0 :
            scores.append(make_model(depth=None, graph=True, data_set=2))
            print("Average accuracy for unconstrained depth : "+ str(np.mean(scores)) +
                  ",\nstandard deviation : "+str(np.std(scores)))
        else:
            scores.append(make_model(depth=i, data_set=2, graph=True))
            print("Average accuracy for max_depth = "+str(i)+" : "+ str(np.mean(scores))+
                  ",\nstandard deviation : "+str(np.std(scores)))
    

