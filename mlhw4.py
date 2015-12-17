# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 04:07:00 2015

@author: Murali
"""

import random as rd
import numpy as np
import operator
import itertools


def reading_data(data_location):
    ls=[]
    with open(data_location,"r") as f:
        ls.append(f.readlines())
    ls=ls[0]
    #print ls
    for i in range(len(ls)):
        ls[i]=ls[i].split()
        ls[i][0]=int(ls[i][0])
        for j in range(1,len(ls[i])):
            ls[i][j]=float(ls[i][j][ls[i][j].find(":")+1:])
    return np.array(ls)

sign= lambda x:-1 if x<0 else 1

def learning_rate(ini,t,c):
    return ini/(1+(ini*t)/c)
    
def svm(data,num_epochs,ini_r,test_data,C,cv):
    wts=np.array([0]*(data.shape[1]-1))
    initial_r=ini_r
    t=0
    co=0
    for i in range(num_epochs):
        np.random.shuffle(data)
        for j in range(data.shape[0]):
            r=learning_rate(initial_r,t,C)
            dt=data[j,1:]
            label=data[j,0]
            #print r
            if np.dot(wts,dt)*label <= 1:
                wts=wts-r*(wts-(C*label*dt))
            else:
                wts=wts-(r*wts)
            t=t+1   
    for i in range(test_data.shape[0]):
        dt=test_data[i,1:]
        des=test_data[i,0]
        pred=np.dot(dt,wts)*des
        #print pred,des
        #pred=sign(np.dot(dt,wts))
        if pred>0:
           co=co+1 
    if cv==0: 
        if np.linalg.norm(wts)!=0:
           margin= 2/(np.linalg.norm(wts))
        else:
           margin=2/0.001
        return float(co)/test_data.shape[0],margin
    else:
        return float(co)/test_data.shape[0]
#acc=svm(train_data,30,0.15,test_data,10)       
#print acc        


def feature_transformation(data):
    lab=data[:,0]
    #lab[lab==0]=-1
    data=data[:,1:]
    arr=np.zeros((data.shape[0],1))
    fl=False
    for i in range(data.shape[1]):
        for j in range(data.shape[1]):
            tba=np.multiply(data[:,i],data[:,j])
            for k in range(arr.shape[1]):
                if np.array_equal(arr[:,k],tba):
                    #print "here"
                    fl=True
            if fl==False:       
               arr=np.column_stack((arr,tba))
            else:
                fl=False
    arr[:,0]=lab            
    return arr        
    
    
                
def cv(data,list_of_rates,list_of_penalty,num_folds,test_data):
    all_vals=itertools.product(list_of_rates,list_of_penalty)
    #print all_vals
    acc_dict={}
    #m_dict={}
    for i in all_vals:
        partition_len=float(data.shape[0])/num_folds
        #print partition_len        
        p=0
        acc=[]
        #mi=0
        for j in range(num_folds):
            v_test=data[p:p+partition_len]
            v_train=np.concatenate((data[:p],data[p+partition_len:]),axis=0)
            f=svm(v_train,10,i[0],v_test,i[1],1)
            #print f
            acc.append(f)
            p=p+partition_len
        acc_dict[i]=float(sum(acc))/float(len(acc))
        #m_dict[i]=mi
        #print acc
    #print "doing something"    
    best_param=max(acc_dict.iteritems(),key=operator.itemgetter(1))[0]
    print "the average accuracy is " + str(acc_dict[best_param])
    final_acc=svm(data,30,best_param[0],test_data,best_param[1],0)
    return acc_dict,best_param,final_acc            
    
"""
def cv(data,list1,list2,num_folds):
    all_hypers=itertools.product(list1,list2)
    acc_dict={}
    for i in all_hypers:
        p=0
        partition_len=float(data.shape[0])/num_folds
        for j in range(num_folds):
            v_test=data[p:p+partition_len,:]
            v_train=np.concatenate((data[:p,:],data[p+partition_len:,:]),axis=0)
            #print v_train.shape
            #print v_test.shape
            print svm(v_train,10,i[0],v_test,i[1])
            p=p+partition_len
"""
#test_data=reading_data("data0/test0.10")  
data0_train=reading_data("data0/train0.10") 
data0_test=reading_data("data0/test0.10")

astro_orig_train=reading_data("astro/original/train")  
astro_orig_train[:,0][astro_orig_train[:,0]==0]=-1
astro_orig_test=reading_data("astro/original/test")
astro_orig_test[:,0][astro_orig_test[:,0]==0]=-1

astro_scaled_train=reading_data("astro/scaled/train")
astro_scaled_train[:,0][astro_scaled_train[:,0]==0]=-1
astro_scaled_test=reading_data("astro/scaled/test")
astro_scaled_test[:,0][astro_scaled_test[:,0]==0]=-1

asotr_new=feature_transformation(astro_orig_train)
asote_new=feature_transformation(astro_orig_test)
asstr_new=feature_transformation(astro_scaled_train)
asste_new=feature_transformation(astro_scaled_test)

def distance(arr):
    ls=[]
    z=np.zeros(arr.shape[1]-1)
    for i in range(arr.shape[0]):
        #print arr[i,1:]
        ls.append(np.linalg.norm(arr[i,1:]-z))
    return max(ls)

list_of_astrodata=[astro_orig_train,astro_orig_test,astro_scaled_train,astro_scaled_test,asotr_new,asote_new,asstr_new,asste_new]
names=["original_train","original_test","scaled_train","scaled_test","original_transformed_train","original_transformed_test","scaled_transformed_train","scaled_transformed_test"]
print "Experiment-2"
for i in range(len(list_of_astrodata)):
    print "The maximum distance from origin in the "+str(names[i])+" data is "+str(distance(list_of_astrodata[i]))
print "--------------------------------------------------------------------"

list_of_astrodata=[astro_orig_train,astro_orig_test,astro_scaled_train,astro_scaled_test,asotr_new,asote_new,asstr_new,asste_new,data0_train,data0_test]
names=["original_train","original_test","scaled_train","scaled_test","original_transformed_train","original_transformed_test","scaled_transformed_train","scaled_transformed_test","data0_train","data0_test"]

print "Experiment-3"
""""
for i in range(0,len(list_of_astrodata),2):
    acc_d,b,test_acc=cv(list_of_astrodata[i],[0.001,0.01,0.1,0.125,1.0],[0.001,0.01,0.1,1,10,30],10,list_of_astrodata[i+1])
    print "The best parameters of "+names[i]+" data: Learning rate is "+str(b[0])+" and penalty term C is "+str(b[1])    
    print "The margin of the weight vector on "+names[i+1]+" data using best hyperparameters is "+ str(test_acc[1])    
    print "The accuracy on the "+names[i+1]+" data using best hyperparameters is "+str(test_acc[0])
    print "----------------------------------------------------------------------"      
"""
#acc_d,b,test_acc=cv(astro_orig_train,[0.125,1.0],[10,20],10,astro_orig_test)

acc_d,b,test_acc=cv(astro_orig_train,[0.001,0.01,0.125,1.0],[0.1,1,10,20,30],10,astro_orig_test)
print "The best parameters on "+ "original_train"+" data: Learning rate is "+str(b[0])+" and penalty term C is "+str(b[1])    
print "The margin of the weight vector on "+"original_train"+" data using best hyperparameters is "+ str(test_acc[1])    
print "The accuracy on the "+"original_test"+" data using best hyperparameters is "+str(test_acc[0])
print "----------------------------------------------------------------------"
acc_d,b,test_acc=cv(astro_scaled_train,[0.001,0.01,0.125,1.0],[0.1,1,10,20,30],10,astro_scaled_test)
print "The best parameters on "+ "scaled_train"+" data: Learning rate is "+str(b[0])+" and penalty term C is "+str(b[1])    
print "The margin of the weight vector on "+"scaled_train"+" data using best hyperparameters is "+ str(test_acc[1])    
print "The accuracy on the "+"scaled_test"+" data using best hyperparameters is "+str(test_acc[0])
print "----------------------------------------------------------------------"

acc_d,b,test_acc=cv(asotr_new,[0.001,0.01,0.125,1.0],[0.1,1,10,20,30],10,asote_new)
print "The best parameters on "+ "original_transformed_train"+" data: Learning rate is "+str(b[0])+" and penalty term C is "+str(b[1])    
print "The margin of the weight vector on "+"original_transformed_train"+" data using best hyperparameters is "+ str(test_acc[1])    
print "The accuracy on the "+"original_transformed_test"+" data using best hyperparameters is "+str(test_acc[0])
print "----------------------------------------------------------------------"

acc_d,b,test_acc=cv(asstr_new,[0.001,0.01,0.125,1.0],[0.1,1,10,20,30],10,asste_new)
print "The best parameters on "+ "scaled_transformed_train"+" data: Learning rate is "+str(b[0])+" and penalty term C is "+str(b[1])    
print "The margin of the weight vector on "+"scaled_transformed_train"+" data using best hyperparameters is "+ str(test_acc[1])    
print "The accuracy on the "+"scaled_transformed_test"+" data using best hyperparameters is "+str(test_acc[0])
print "----------------------------------------------------------------------"

acc_d,b,test_acc=cv(data0_train,[0.001,0.01,0.125,1.0],[0.1,1,10,20,30],10,data0_test)
print "The best parameters on "+ "data0"+" data: Learning rate is "+str(b[0])+" and penalty term C is "+str(b[1])    
print "The margin of the weight vector on "+"scaled_transformed_train"+" data using best hyperparameters is "+ str(test_acc[1])    
print "The accuracy on the "+"data0"+" data using best hyperparameters is "+str(test_acc[0])
print "----------------------------------------------------------------------"
"""
Experiment-3
the average accuracy is 0.81357857353
The best parameters on original_train data: Learning rate is 0.125 and penalty term C is 1
The margin of the weight vector on original_train data using best hyperparameters is 33.5473134776
The accuracy on the original_test data using best hyperparameters is 0.7705

the average accuracy is 0.805477451351
The best parameters on scaled_train data: Learning rate is 1.0 and penalty term C is 30
The margin of the weight vector on scaled_train data using best hyperparameters is 0.892258801456
The accuracy on the scaled_test data using best hyperparameters is 0.7725

the average accuracy is 0.833980582524
The best parameters on original_transformed_train data: Learning rate is 0.001 and penalty term C is 0.1
The margin of the weight vector on original_transformed_train data using best hyperparameters is 58.7841627188
The accuracy on the original_transformed_test data using best hyperparameters is 0.835

the average accuracy is 0.798654015887
The best parameters on scaled_transformed_train data: Learning rate is 1.0 and penalty term C is 30
The margin of the weight vector on scaled_transformed_train data using best hyperparameters is 0.785701586013
The accuracy on the scaled_transformed_test data using best hyperparameters is 0.79975

the average accuracy is 1.0
The best parameters on data0 data: Learning rate is 0.001 and penalty term C is 30
The margin of the weight vector on scaled_transformed_train data using best hyperparameters is 0.516213120404
The accuracy on the data0 data using best hyperparameters is 1.0
"""
#print "----------------------------------------------------------------------"    
