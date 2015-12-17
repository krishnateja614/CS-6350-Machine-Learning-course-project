# -*- coding: utf-8 -*-
"""
Created on Tue Dec 01 00:16:36 2015

@author: Murali
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 18:57:39 2015

@author: Murali
"""

#structure
#1. create a decision tree data structure
#def decision_tree(data,input_features,output_label,heuristic):
import argparse
import collections
import os
import operator
import math
import numpy as np

def most_freq_label(tr_data,target_label_index):
    if target_label_index>= tr_data.shape[1]:
        print "Column index is out of bounds"
    else:
       num_ones=sum([1 for i in tr_data[:,target_label_index] if i==1])
       num_zeros=tr_data.shape[0]-num_ones
       return max(num_zeros,num_ones)        

def unique_values(data,feature_index):
    return list(set(data[:,feature_index]))

def get_data_at_a_feature(data,feature_index,feature_value):
    if feature_index >= data.shape[1]:
        print "Index out of bounds"
    else:    
        return data[data[:,feature_index]==feature_value],np.delete(data,feature_index,axis=1)
    
def best_feature(data,target_label,gain_function):
    best_feat=0.0
    most_gain=0.0
    #target_feature_data=[]
    for i in range(data.shape[1]):
        g=gain_function(data,i,target_label)
        if (g>most_gain and i!= target_label):
            most_gain=g
            best_feat=i
    return best_feat        

def entropy(data,target_label):
    #target_feature_data=[]
    tot_entropy=0.0
    target_feature_data=data[:,target_label]    
    l=data.shape[0]
    u=unique_values(data,target_label)
    for i in u:
       count=sum([1 for j in target_feature_data if j==i])  
       tot_entropy+= (-float(count)/float(l))*((math.log(float(count)/float(l),2)))
       
    return tot_entropy   

def gain_function(data,feature,target_label):
    gain=0.0
    exp_entropy=0.0
    u=unique_values(data,feature)
    total_len=data.shape[0]
    for i in u:
        subset_data=get_data_at_a_feature(data,feature,i)[0]
        sub_len=subset_data.shape[0]
        exp_entropy+=float(sub_len)/float(total_len)*entropy(subset_data,target_label)
    gain=entropy(data,target_label)-exp_entropy     
    return gain

def decision_tree_ds(data,height,target_label,gain_function):  
   
    print "Learning and constructing the tree" 
    #target_data=data[:,target_label]
    majority_label=most_freq_label(data,target_label)     
    print "hi"
    best=best_feature(data,target_label,gain_function)
    tree = {best:collections.defaultdict(lambda: majority_label)}
    if height<=0:
        return tree
    if height>0:
       majority_label=most_freq_label(data,target_label)     
       print "hi"
       best=best_feature(data,target_label,gain_function)
       tree = {best:collections.defaultdict(lambda: majority_label)}
       for i in unique_values(data,best):
              #if h<=height:
           sub_tree=decision_tree_ds(get_data_at_a_feature(data,best,i)[1],height-1,target_label-1,gain_function)
           tree[best][i]=sub_tree
            #h=h+1
           return tree    
   
                 
    
    
def rf_classification_func(datapoint,tree):
    if tree=="positive" or tree=="negative":
        return tree      
    else:
        feature = tree.keys()[0]
        t = tree[feature][datapoint[feature]]
        return rf_classification_func(datapoint, t)

def rf_full_classification(test_data,tree):
    res=[]
    #print "classifying"
    for i in range(test_data.shape[0]):
        res.append(rf_classification_func(test_data[i],tree))
    return res    
            
    
if __name__=="__main__":
   #parser = argparse.ArgumentParser(description='This program takes in the destination directory of training and test data sets as parameters and creates and trains a decision tree using ID3')
   # 
   #parser.add_argument('-d', dest='path2datafile', default="C:/Users/Murali/Downloads/tic-tac-toe/", type=str,
   #                    help='Specifies the directory of training and testsets which is used by the decision tree')
   #args = parser.parse_args()
   #location="tic-tac-toe/"
   #x,y=preparing_data(location)
   #target_values=[]
   #for i in range(len(y)):
   #    target_values.append(y[i]["target"])
   #for i in range(len(y)):
   #    del y[i]['target']
       
   #feature_list=["C0","C1","C2","C3","C4","C5","C6","C7","C8","target"]
   #target_label="target"
   #decision_tree=decision_tree_ds(x,2,x.shape[1] - 1,gain_function)
   
  # predicted_values=full_classification(y,decision_tree)
   #acc=0
   #for i in range(len(target_values)):
   #    if predicted_values[i]==target_values[i]:
   #        acc+=1
   #print "The Accuracy of the decision tree is " + str(float(acc)/float(len(target_values)))        
   print "hi"
