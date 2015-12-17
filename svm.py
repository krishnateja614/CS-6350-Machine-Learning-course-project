# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 23:25:12 2015

@author: Murali
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 04:07:00 2015

@author: Murali
"""

import random as rd
import numpy as np
import operator
import itertools

#sign= lambda x:-1 if x<0 else 1

class SVM(object):
    def __init__(self,data,learning_rate,reg_parameter):
        data[:,0][data[:,0]==0]=-1.0
        self.data=data
        self.wts=np.array([0]*(data.shape[1]-1))
        self.learning_rate=learning_rate
        self.reg_parameter=reg_parameter
        
    def sign(x):
        return 1 if x<0 else 1
        
    def learning_rate_fun(self,t,c):
         return self.learning_rate/(1+(self.learning_rate*t)/c)
         
    def sgd(self,num_epochs):
        for i in range(num_epochs):
           t=0
           np.random.shuffle(self.data)
           for j in range(self.data.shape[0]):
               r=self.learning_rate_fun(t,self.reg_parameter)
               dt=self.data[j,1:]
               label=self.data[j,0]
            #print r
               if np.dot(self.wts,dt)*label <= 1:
                  self.wts=self.wts-r*(self.wts-(self.reg_parameter*label*dt))
               else:
                   self.wts=self.wts-(r*self.wts)
               t=t+1
               
    def svm_train(self,num_epochs):
        self.sgd(num_epochs)
    
    def svm_predict(self,test_data):
        #self.wts=self.wts.reshape(test_data.shape[1],1)
        pred=np.dot(test_data,self.wts)
        pred=list(pred)
        svm_preds=[0 if i<0 else 1 for i in pred]
        return svm_preds
        
        


                
def svm_cv(data,list_of_rates,list_of_penalty,num_folds,test_data):
    all_vals=itertools.product(list_of_rates,list_of_penalty)
    acc_dict={}
    
    for i in all_vals:
        partition_len=float(data.shape[0])/num_folds
        #print partition_len        
        p=0
        ls=[]
        #mi=0
        for j in range(num_folds):
            v_test=data[p:p+partition_len]
            v_train=np.concatenate((data[:p],data[p+partition_len:]),axis=0)
            f=SVM(v_train,i[0],i[1])
            f.svm_train(50)
            preds=f.svm_predict(v_test[:,1])
            #print f
            ls.append(acc(preds,v_test[:,0]))
            p=p+partition_len
        acc_dict[i]=float(sum(ls))/float(len(ls))
        #m_dict[i]=mi
        #print acc
    #print "doing something"    
    best_param=max(acc_dict.iteritems(),key=operator.itemgetter(1))[0]
    print "the average accuracy is " + str(acc_dict[best_param])
    final_model=SVM(data,best_param[0],best_param[1])
    final_model.svm_train(50)
    final_preds=final_model.svm_predict(test_data[:,1:])
    return final_preds          
    
def acc(ls_1,ls_2):
    c=0
    if len(ls_1)!=len(ls_2):
        print "not equal length lists"
    for i in range(len(ls_1)):
        if ls_1[i]==ls_2[i]:
            c+=1
    return float(c)/len(ls_1)      
    
    
