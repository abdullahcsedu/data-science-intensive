#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score



### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
###########clf = SVC(kernel ="linear")
features_train = features_train[:len(features_train)/100] 
labels_train = labels_train[:len(labels_train)/100] 
clf = SVC(C =10000.0 ,kernel ='rbf')
start_time_train = time()
clf.fit(features_train ,labels_train ) 
end_time_train =  time() 
print "Training Time rbf  " , round (end_time_train - start_time_train  , 3 )
#########################################################
start_time_pred = time()
pred = clf.predict(features_test)
end_time_pred =  time() 
print "Prediction Time  rbf" , round (end_time_pred - start_time_pred  , 3 )
answer = pred[9]
## o or 1 for which class
print pred[10] ,  pred[26] , pred[50]
accuracy = accuracy_score(  pred  , labels_test )
print "accuracy for rbf " , accuracy



###clf1 = SVC(kernel ="linear")
###start_time_train = time()
###clf1.fit(features_train ,labels_train ) 
###end_time_train =  time() 
####print "Training Time For linear " , round (end_time_train - start_time_train  , 3 )
#########################################################
###start_time_pred = time()
###pred = clf1.predict(features_test)
###end_time_pred =  time() 
###print "Prediction Time  for linear" , round (end_time_pred - start_time_pred  , 3 )
####accuracy = accuracy_score(  pred  , labels_test )
###print "accuracy is" , accuracy

