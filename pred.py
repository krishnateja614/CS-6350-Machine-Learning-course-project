# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 19:18:26 2015

@author: Murali
"""
import itertools
import numpy as np
import csv
import os
import json
import pandas as pd
import re
import textblob as tb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from logisticregression import LogisticRegression as lr
from sklearn import metrics,pipeline,grid_search
from svm import SVM
import operator
from nearestneighbours import knn


stopwords = ['a', 'able', 'about', 'across', 'after', 'all', 'almost', 'also', 'am', 'among',
	     'an','and','any','are','as','at','be','because','been','but','by','can',
             'cannot','could','dear','did','do','does','either','else','ever','every',
             'for','from','get','got','had','has','have','he','her','hers','him','his',
             'how','however','i','if','in','into','is','it','its','just','least','let',
             'like','likely','may','me','might','most','must','my','neither','no','nor',
             'not','of','off','often','on','only','or','other','our','own','rather','said',
             'say','says','she','should','since','so','some','than','that','the','their',
             'them','then','there','these','they','this','tis','to','too','twas','us',
             'wants','was','we','were','what','when','where','which','while','who',
             'whom','why','will','with','would','yet','you','your','emptyempty','http','www','com','html','url']
             

def reading_data(data):
    #ls=[]    
    num_ls=[]
    text_ls=[]
    with open(data,"r") as f:
        tsvreader = csv.reader(f, delimiter="\t")
        #ls.append(tsvreader.readline)
        i=0
        for line in tsvreader:
            if i==0:
               #ls.append(line[:2]+line[3:]+["body","title"])
               num_ls.append([line[1]]+line[4:])
               text_ls.append([line[0]]+[line[3]]+["body","title"])
               i=i+1   
            else:
                cur_dct=json.loads(line[2])
                temp=[]
                for j in cur_dct.keys():
                    if j!="url":
                       temp.append(cur_dct[j])
                #print i       
                num_ls.append([float(k) if k!="?" else 0 for k in [line[1]]+line[4:]])
                text_ls.append([line[0]]+[line[3]]+temp)
    return num_ls,text_ls            
    
    
train_num,train_text=reading_data(r"F:\Kaggle\Stumbleupon evergreen Classification\train.tsv")
test_num,test_text=reading_data(r"F:\Kaggle\Stumbleupon evergreen Classification\test.tsv")

train_num=train_num[1:]
test_num=test_num[1:]
train_text=train_text[1:]
test_text=test_text[1:]
#feature-1 : The least year if any in the webpage and make it a categorical value/continous
#regex example ->  x=re.findall(r'\b[1-9]\d{3}\b',"i am 0012 and 19000 and 1900") 
#x=['1900']
#re.findall(r'[1-9]\d{3}\w*',"i am 0012 and 19000 and 1900 and 3003u and u1399")
#['19000', '1900', '3003u', '1399'] .clean this by looping over the list and check if the length is >4,
#if yes,iterate from the last of the string and check if this is alphabet and if by its removal if we get
#a 4 digit year, remove it else if its a number delete it from the list

alphabets=["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]

def latest_year_and_counts_of_years(data):
    ls=re.findall(r'[1-9]\d{3}\w*',data)
    for i in ls:
        if len(i)>4:
            if i[-1] in alphabets:
                i=int(i[:-1])
            else:
                ls.remove(i)
        else:
            i=int(i)
    return sorted(ls)[0],len(ls)

does_title_has_number=lambda data: re.sub('[^1-9]',"",data)!=""
length = lambda x: len(x.split())
    
    
def clean_merge_data(data):
    ls=[]
    for i in range(len(data)):
        s=str()
        for j in range(len(data[i])):
            #repr(train_text[1][2]).replace(r"\u"," ")
           if str(type(data[i][j])).find("str")!=-1:
              s=s+re.sub(r'[^a-z]', ' ', repr(data[i][j].lower()).replace(r"\u"," "))
           else:
              s=s+"empty"
        ls.append(s)   
    return ls    

train_text=clean_merge_data(train_text)     
test_text=clean_merge_data(test_text)   


#tf-idf
tfv = TfidfVectorizer(min_df=3,  max_features=None, 
       strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
       ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
       stop_words = stopwords)         
"""        
feature_set1 = TfidfVectorizer(min_df=3,  max_features=None, 
               strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}', 
               use_idf=0,sublinear_tf=1, stop_words = 'english')   
feature_set2 = TfidfVectorizer(min_df=3,  max_features=None, 
               strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}', 
               use_idf=1,smooth_idf=1,sublinear_tf=1, stop_words = 'english')      
feature_set3 = TfidfVectorizer(min_df=3,  max_features=None, 
               strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}', 
               ngram_range=(1,4),use_idf=0,sublinear_tf=1, stop_words = 'english') 
feature_set4 = TfidfVectorizer(min_df=3,  max_features=None, 
               strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}', 
               ngram_range=(1,4),use_idf=1,smooth_idf=1,sublinear_tf=1, stop_words = 'english')                    
"""
full_text=np.array(train_text+test_text)
print "Doing TF-IDF transformation"
tfv.fit(full_text)
full_text_tfidf=tfv.transform(full_text)
#train_text=full_text_tfidf[:len(train_text)]
#test_text=full_text_tfidf[len(train_text):]
#svd
svd = TruncatedSVD(algorithm='randomized', n_components=350, n_iter=5,
          random_state=None, tol=0.0) 
print "SVD"
svd.fit(full_text_tfidf)
full_text_svd=svd.transform(full_text_tfidf)
 
train_text=full_text_svd[:len(train_text)]
test_text=full_text_svd[len(train_text):]

train_num=np.array(train_num)
y=train_num[:,-1]

#np.hstack combine svd output 400 columns to the original features in train_num set
if __name__=="__main__":
   """ 
   rf_model=RandomForestClassifier()
#scl = StandardScaler(with_mean=False)
#svm_model = SVC()
   clf = pipeline.Pipeline([('rf', rf_model)])
   param_grid = {'rf__n_estimators': [150,200],'rf__max_depth':[5,7,None],'rf__min_samples_leaf':[2,1],'rf__min_samples_split':[3,2],'rf__max_features':['auto','log2',None]}
   auc_scorer=metrics.make_scorer(roc_auc_score,greater_is_better=True)
   g_model = grid_search.GridSearchCV(estimator = clf, param_grid=param_grid, scoring=auc_scorer,
                                     verbose=10,n_jobs=4, iid=True, refit=True, cv=4)

   g_model.fit(train_text,y)
   print g_model.best_score
   best_parameters = g_model.best_estimator_.get_params()
   for param_name in sorted(param_grid.keys()):
       print("\t%s: %r" % (param_name, best_parameters[param_name]))
    
   best_model = g_model.best_estimator_
 
   best_model.fit(train_text,y)
   rf_preds = best_model.predict(test_text)
   """

      
def cross_validation_lr(X,Y,list_of_learning,list_of_reg,num_cv,test_data):
    #Y=Y.reshape(Y.shape[0],1)
    acc_dict={}
    Y=list(Y)
    partition_len=int(float(X.shape[0])/num_cv)
    all_hyp=itertools.product(list_of_learning,list_of_reg)
    for j in all_hyp:
        i=0
        ls=[]
        for k in range(num_cv):
            te=X[i:i+partition_len,:]
            tr=np.append(X[:i,:],X[i+partition_len:,:],axis=0)
            y_tr=Y[:i]+Y[i+partition_len:]
            y_ts=Y[i:i+partition_len]
            model=lr(tr,np.array(y_tr),j[0],j[1])
            model.train(50)
            pr=model.predict(te)
            ls.append(roc_auc_score(y_ts,pr))
            i=i+partition_len
        ls=ls[:-1]
        avg_acc=float(sum(ls))/len(ls)
        acc_dict[j]=avg_acc
    best_param=max(acc_dict.iteritems(),key=operator.itemgetter(1))[0]
    print best_param
    final_model=lr(X,np.array(Y),best_param[0],best_param[1])
    final_model.train(50)
    preds=final_model.predict(test_data) 
    return preds

def knn_cv(X,Y,list_of_k,num_cv,test_data):
    acc_dict={}
    Y=list(Y)
    partition_len=int(float(X.shape[0])/num_cv)
    all_hyp=list_of_k
    for j in all_hyp:
        i=0
        ls=[]
        for k in range(num_cv):
            te=X[i:i+partition_len,:]
            tr=np.append(X[:i,:],X[i+partition_len:,:],axis=0)
            y_tr=Y[:i]+Y[i+partition_len:]
            y_ts=Y[i:i+partition_len]
            model=knn(tr,y_tr,te)
            pr=model.knn_predict(j)
            print roc_auc_score(y_ts,pr)
            ls.append(roc_auc_score(y_ts,pr))
            i=i+partition_len
        ls=ls[:-1]
        avg_acc=float(sum(ls))/len(ls)
        acc_dict[j]=avg_acc
    best_param=max(acc_dict.iteritems(),key=operator.itemgetter(1))[0]
    print best_param
    final_model=knn(X,Y,test_data)
    preds=final_model.knn_predict(best_param) 
    return preds

    
    
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
            preds=f.svm_predict(v_test[:,1:])
            print acc(preds,v_test[:,0])
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
    final_preds=final_model.svm_predict(test_data)
    return final_preds            
      
def acc(ls_1,ls_2):
    c=0
    if len(ls_1)!=len(ls_2):
        print "not equal length lists"
    for i in range(len(ls_1)):
        if ls_1[i]==ls_2[i]:
            c+=1
    return float(c)/len(ls_1)      

"""
testfile = pd.read_csv('F:/Kaggle/Stumbleupon evergreen Classification/test.tsv', sep="\t", na_values=['?'], index_col=1)
pred_df = pd.DataFrame(preds_knn, index=testfile.index, columns=['label'])
pred_df.to_csv('benchmark.csv')
"""    