# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 19:55:34 2015

@author: Murali
"""

import numpy as np
from math import *
import itertools
#from sklearn.metrics import roc_auc_score

class LogisticRegression(object):
    
    def __init__(self,X,Y,learning_rate,reg_parameter):
        x=(X-np.mean(X))/np.std(X)
        col_1=np.array([1.0]*x.shape[0]).reshape(x.shape[0],1)
        self.X=np.append(col_1,x,axis=1)
        self.Y=Y
        self.learning_rate=learning_rate
        self.reg=reg_parameter
        self.theta=np.array([0.0]*(x.shape[1]+1))
         
    def sigmoid(self,theta,x):
        return 1.0/(1.0+np.exp(-x.dot(theta)))
        
    def cost_func(self):
        h = self.sigmoid(self.theta,self.X)
        #print h.shape
        cost_1 = (-1.0) * self.Y * np.log(h)
        cost_0 = (-1.0) * ((1.0 - self.Y) * np.log(1.0 - h))
        cost = 1.0/self.X.shape[0] *(sum(cost_1 + cost_0) + 0.5 * self.reg * sum(self.theta[1:] ** 2))
        return cost
    
    def gradient_descent(self,num_iters):
        #ls=[]
        prev=1.0
        for i in range(num_iters):           
            temp=self.theta
            pred=self.sigmoid(temp, self.X) 
            error=pred- self.Y
            self.theta[0]=temp[0]-self.learning_rate * 1.0/self.X.shape[0] * sum(error*self.X[:,0])
            for j in range(1,self.X.shape[1]):
                t=temp[j]-self.learning_rate * (1.0/self.X.shape[0] * sum(error*self.X[:,j]) + (self.reg *  1.0/self.X.shape[0] * temp[j]))
                self.theta[j]=t 
            if prev-self.cost_func()>0.00001:
               prev=self.cost_func()
            else:
                break
            #print self.cost_func()
        #return ls    

    def train(self,num_iters):
        return self.gradient_descent(num_iters)
    
    def predict(self,data):
        data=(data-np.mean(data))/np.std(data)
        col_1=np.array([1.0]*data.shape[0]).reshape(data.shape[0],1)
        data=np.append(col_1,data,axis=1)
        preds=self.sigmoid(self.theta,data)
        return list(preds)
    
def cross_validation_lr(X,Y,list_of_learning,list_of_reg,num_cv):
    #Y=Y.reshape(Y.shape[0],1)
    Y=list(Y)
    partition_len=int(float(X.shape[0])/num_cv)
    all_hyp=itertools.product(list_of_learning,list_of_reg)
    for j in all_hyp:
        i=0
        for k in range(num_cv):
            te=X[i:i+partition_len,:]
            tr=np.append(X[:i,:],X[i+partition_len:,:],axis=0)
            y_tr=Y[:i]+Y[i+partition_len:]
            y_ts=Y[i:i+partition_len]
            model=LogisticRegression(tr,y_tr,j[0],j[1])
            model.train(10)
            pr=model.predict(te)
            print acc(pr,y_ts)
            i=i+partition_len
            
def acc(ls_1,ls_2):
    c=0
    if len(ls_1)!=len(ls_2):
        print "not equal length lists"
    for i in range(len(ls_1)):
        if ls_1[i]==ls_2[i]:
            c+=1
    return float(c)/len(ls_1)        
        
    
    