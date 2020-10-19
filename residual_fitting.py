"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classical algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

from data import make_data1, make_data2
from plot import plot_boundary, plot_boundary_extended
from scipy import stats
from sklearn.metrics import accuracy_score


class residual_fitting(BaseEstimator, ClassifierMixin):
    
    w = []
    pred = []
    prediction = []
    proba = []

    def fit(self, X, y):
        """Fit a Residual fitting model using the training set (X, y).

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        # Input validation
        X = np.asarray(X, dtype=np.float)
        if X.ndim != 2:
            raise ValueError("X must be 2 dimensional")

        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("The number of samples differs between X and y")

        nbr_att = 5 ;
        size_ts = 10000
        size_ls = 250
        
        residus = []
        step = 0
        
        self.w.append(np.mean(y))
        print("weight 0 = "+str(self.w[0]))
    
        for att in range(1,nbr_att+1) :
            
            sum_res = 0 
            for k in range(1,step) :
                sum_res += np.array(X[:,k-1])*self.w[k]
                
            residus.append(y - self.w[0] - sum_res)
            # print(residus[att-1])
            self.w.append( (stats.pearsonr(X[:,att-1],residus[att-1])[0]) * (np.std(residus[att-1])) )
            # print("mean = "+str(np.mean(residus[att-1])))
            print("weight "+str(att)+" = "+str(self.w[att]))
            step += 1

        return self

    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes, or the predict values.
        """
        
        nbr_att = 5
        sum_res = 0 

        for att in range(nbr_att) :
            sum_res +=  np.array(X[:,att])*self.w[att+1]
            
        self.pred = self.w[0] + sum_res     
        
        lim = 0
        for i in range(nbr_att) :
            lim += self.w[i]**2
            
        print("limite = "+str(lim))
        
        for i in range(10000) :
            self.pred[i] = self.pred[i]**2
            if self.pred[i] >= lim :
                self.prediction.append(1)
            else:
                self.prediction.append(0)

        return self.prediction

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """
        
        siz = len(X)
        
        self.proba = np.zeros(shape=(len(X),2))

        nbr_att = 5

        lim = 0
        for i in range(nbr_att) :
            lim += self.w[i]**2
            
        maxi = max(self.pred)
        maxi -= lim
        
        for i in range(siz):
            if self.prediction[i] == 0:
                prob1 = 1 - ((lim - self.pred[i])/lim)/2
                prob2 = 1 - prob1
                self.proba[i,0] = prob1
                self.proba[i,1] = prob2
            else:
                prob2 = 0.5 + ((self.pred[i] - lim)/maxi)/2
                prob1 = 1 - prob2
                self.proba[i,0] = prob1
                self.proba[i,1] = prob2

        return self.proba
    
    def add_attributes(self, X_train, X_test):
        
        size_ts = 10000
        size_ls = 250
        
        modified_X_train = []
        modified_X_test = []
        
        X1X1_X_train = []
        X2X2_X_train = []
        X1X2_X_train = []
        
        for i in range(size_ls) :
            X1X1_X_train.append(X_train[i,0]**2)
            X2X2_X_train.append(X_train[i,1]**2)
            X1X2_X_train.append(X_train[i,0]*X_train[i,1])
           
        modified_X_train = np.column_stack((X_train[:,0], X_train[:,1] ,
                                           X1X1_X_train, X2X2_X_train, X1X2_X_train))
            
        X1X1_X_test = []
        X2X2_X_test = []
        X1X2_X_test = []
        
        for i in range(size_ts) :
            X1X1_X_test.append(X_test[i,0]**2)
            X2X2_X_test.append(X_test[i,1]**2)
            X1X2_X_test.append(X_test[i,0]*X_test[i,1])
            
        modified_X_test = np.column_stack((X_test[:,0], X_test[:,1] ,
                                           X1X1_X_test, X2X2_X_test, X1X2_X_test))
        
        return modified_X_train, modified_X_test
            

if __name__ == "__main__":
    
    size_ts = 10000
    size_ls = 250
    
    [X_train, y_train, X_test, y_test] = make_data1(size_ts,size_ls,0,random_state=0)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    clf = residual_fitting()
    
    [X_train_add, X_test_add] = clf.add_attributes(X_train,X_test)
    
    # clf.fit(X=X_train, y=y_train)
    clf.fit(X=X_train_add, y=y_train)
    
    # clf.predict(X_test)
    clf.predict(X_test_add)
    
    clf.predict_proba(X_test)
    
    plot_boundary_extended(fname="data2",fitted_estimator=clf, X=X_test, y=y_test)
    
    print("Accuracy score = "+str(accuracy_score(y_test, clf.prediction))) 