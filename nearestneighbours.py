# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 22:11:12 2015

@author: Murali
"""

import numpy as np
import operator

class knn(object):
    
    def __init__(self,train_data,labels,test_data):
        #print "Make sure label is the last column in training data"
        self.train=train_data
        self.test=test_data
        self.labels=labels
        
    def knn_predict(self,k):
        preds=[]         
        for i in self.test:
            closest_k={}
            for j in range(len(self.train)):
                closest_k[j]=np.linalg.norm(i-self.train[j])
            top_k_elements=sorted(closest_k.iteritems(),key=operator.itemgetter(1))[:k]
            top_k_indices=[m[0] for m in top_k_elements]
            top_k_preds=[self.labels[n] for n in top_k_indices]
            preds.append(float(sum(top_k_preds))/len(top_k_preds))  
        return preds    
            
            
        
        
        