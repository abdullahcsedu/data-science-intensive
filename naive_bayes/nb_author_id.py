#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
import numpy
from sklearn.naive_bayes import GaussianNB
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
clf = GaussianNB()
start_time_train = time()
clf.fit(features_train ,labels_train ) 
end_time_train =  time() 
print "Training Time " , round (end_time_train - start_time_train  , 3 )

start_time_pred = time()
pred = clf.predict(features_test)
end_time_pred =  time() 
print "Prediction Time " , round (end_time_pred - start_time_pred  , 3 )
accuracy = accuracy_score(  pred  , labels_test )
#########################################################

print "accuracy is" , accuracy
 

