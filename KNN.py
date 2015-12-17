# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 04:03:37 2015

@author: Murali
"""
import collections
import operator
import argparse

 
def nearest_neighbours(train_data,test_data,k):
    res=[]
    target_label=[i["target"] for i in test_data]
    for i in range(len(test_data)):
        dct={}
        for j in range(len(train_data)):
            dct[j]=hamming_distance(test_data[i]["current_position"],train_data[j]["current_position"])
        list_of_k_indices=[m for (m,n) in sorted(dct.iteritems(), key=operator.itemgetter(1))[:k]]
        fin_dct={}
        for new_iter in list_of_k_indices:
            if train_data[new_iter]["target"] in fin_dct:
                fin_dct[train_data[new_iter]["target"]]+=1
            else:
                fin_dct[train_data[new_iter]["target"]]=1
        res.append(max(fin_dct.iteritems(),key=operator.itemgetter(1))[0])
    acc=0    
    for i in range(len(res)):
        if res[i]==target_label[i]:
            acc+=1
    return float(acc)/len(target_label)        
        
def preparing_data(list_of_data):
    
    #_data=[train_1,train_2,train_3,train_4,train_5,train_6]
    data=[]
    for i in list_of_data:
        with open(i,"r") as f:
            for j in f.readlines():
                 data_ls=[]
                 temp=j.replace("\n","").split(",")
                 tmp="".join([str(i) for i in temp[:-1]])
                 data_ls.append(tmp)
                 data_ls.append(temp[-1])
                 
                 data.append(data_ls)
    
    #test_1=datalocation+"tic-tac-toe-test.txt"   
    #test_data=[]
    #with open(test_1,"r") as f:
    #     for i in f.readlines():
    #         test_data.append(i.replace("\n","").split(","))
             
    #if type_of_data=="train":
    feature_names=["current_position","target"]
    #elif type_of_data=="test"
    #    feature_names=["current_position"]
    data_dict=[]
    for line in data:
        data_dict.append(dict(zip(feature_names,line)))
    #for line in test_data:
    #    test_data_dict.append(dict(zip(test_feature_names,line)))    
    return data_dict

def cv(datalocation,folds,hyper_param_list):
    
    train_1=datalocation+"tic-tac-toe-train-1.txt"
    train_2=datalocation+"tic-tac-toe-train-2.txt"
    train_3=datalocation+"tic-tac-toe-train-3.txt"
    train_4=datalocation+"tic-tac-toe-train-4.txt"
    train_5=datalocation+"tic-tac-toe-train-5.txt"
    train_6=datalocation+"tic-tac-toe-train-6.txt"
    
    list_of_train_data=[train_1,train_2,train_3,train_4,train_5,train_6]
    acc_dict={}
    for i in hyper_param_list:
        acc=[]
        #c=0
        for j in range(folds):
            #c=c+1
            test=[list_of_train_data[j]]
            train=list_of_train_data[:j]+list_of_train_data[j+1:]
            #print train
            #print test
            train_dict=preparing_data(train)
            test_dict=preparing_data(test)
            #print train_dict
            #print test_dict
            
            acc.append(nearest_neighbours(train_dict,test_dict,i,hamming_distance))
        #print c        
        print "When Hyper parameter is " + str(i)+", the CV accuracy values for each fold are"
        print acc
        acc_dict[i]=float(sum(acc))/float(len(acc))
    
    best_k=max(acc_dict.iteritems(),key=operator.itemgetter(1))[0]
    test_1=datalocation+"tic-tac-toe-test.txt"   
    fin_test_data=[test_1]
    fin_test_data_dict=preparing_data(fin_test_data)
    fin_train_data_dict=preparing_data(list_of_train_data)
    best_acc=nearest_neighbours(fin_train_data_dict,fin_test_data_dict,best_k,hamming_distance)
    
    return acc_dict,best_k,best_acc    
    
    
if __name__=="__main__":
   #parser = argparse.ArgumentParser(description='This program takes in the destination directory of training and test data sets as parameters and does CV, KNN')
   # 
   #parser.add_argument('-d', dest='path2datafile', default="C:/Users/Murali/Downloads/tic-tac-toe/", type=str,
   #                    help='Specifies the directory of training and testsets which is used by the decision tree')
   #args = parser.parse_args()
   location="tic-tac-toe/"  
   x,y,z=cv(location,6,[1,2,3,4,5])
   print "The average accuracy for each value of the hyperparameter from CV is"
   print x
   print "The best value of k from CV is: " + str(y)
   print "Using the best value for k i.e k = "+str(y)+", the accuracy for K-Nearest Neighbours is: " + str(z)
   #x,y=preparing_data(args.path2datafile)  
   #feature_list=["C0","C1","C2","C3","C4","C5","C6","C7","C8","target"]
   #target_label="target"
   #decision_tree=decision_tree_ds(x,feature_list,"target",gain_function)
   #print full_classification(y,decision_tree)    
            
            
            
            
        
            