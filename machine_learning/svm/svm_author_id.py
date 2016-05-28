
# coding: utf-8

# ## Support Vector Machine Mini Project
# 
# Import, create, train and make predictions with the sklearn SVC classifier. When creating the classifier, use a linear kernel (if you forget this step, you will be unpleasantly surprised by how long the classifier takes to train). What is the accuracy of the classifier?

# In[ ]:




# In[4]:

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

#########################################################


clf = SVC(kernel="linear")


#### now your job is to fit the classifier
#### using the training features/labels, and to
#### make a set of predictions on the test data
clf.fit(features_train, labels_train)  
#### store your predictions in a list named pred
pred = clf.predict( features_test)
acc = accuracy_score(pred, labels_test)

print acc






# ## SVM Author ID Timing

# In[ ]:

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

#########################################################


clf = SVC(kernel="linear")


#### now your job is to fit the classifier
#### using the training features/labels, and to
#### make a set of predictions on the test data
t0 = time()
clf.fit(features_train, labels_train)  
t1 = round(time()-t0, 3)
print 'training time : ', t1 
t2 =  time()
#### store your predictions in a list named pred
pred = clf.predict( features_test)
t3 = round(time()-t2, 3)
print 'prediction time : ' ,t3 
    
if t3 > t1 :
    return 'prediction time is higher than training time' ;
else :
    return  'training time is higher than prediction time'


# ## A Smaller Training Set
# 
# One way to speed up an algorithm is to train it on a smaller training dataset. The tradeoff is that the accuracy almost always goes down when you do this. Let’s explore this more concretely: add in the following two lines immediately before training your classifier. 
# 
# features_train = features_train[:len(features_train)/100] 
# labels_train = labels_train[:len(labels_train)/100] 
# 
# These lines effectively slice the training dataset down to 1% of its original size, tossing out 99% of the training data. You can leave all other code unchanged. What’s the accuracy now?

# In[1]:


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


features_train = features_train[:len(features_train)/100] 
labels_train = labels_train[:len(labels_train)/100] 


clf = SVC(kernel="linear")


#### now your job is to fit the classifier
#### using the training features/labels, and to
#### make a set of predictions on the test data
clf.fit(features_train, labels_train)  
#### store your predictions in a list named pred
pred = clf.predict( features_test)
acc = accuracy_score(pred, labels_test)

print acc



# ## Deploy an RBF Kernel
# 
# Keep the training set slice code from the last quiz, so that you are still training on only 1% of the full training set. Change the kernel of your SVM to “rbf”. What’s the accuracy now, with this more complex kernel?

# In[2]:


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


features_train = features_train[:len(features_train)/100] 
labels_train = labels_train[:len(labels_train)/100] 


clf = SVC(kernel="rbf")


#### now your job is to fit the classifier
#### using the training features/labels, and to
#### make a set of predictions on the test data
clf.fit(features_train, labels_train)  
#### store your predictions in a list named pred
pred = clf.predict( features_test)
acc = accuracy_score(pred, labels_test)

print acc


# ## Optimize C Parameter
# 
# Keep the training set size and rbf kernel from the last quiz, but try several values of C (say, 10.0, 100., 1000., and 10000.). Which one gives the best accuracy

# In[1]:

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


features_train = features_train[:len(features_train)/100] 
labels_train = labels_train[:len(labels_train)/100] 


clf = SVC(kernel="rbf" , C= 1.0)


#### now your job is to fit the classifier
#### using the training features/labels, and to
#### make a set of predictions on the test data
clf.fit(features_train, labels_train)  
#### store your predictions in a list named pred
pred = clf.predict( features_test)
acc = accuracy_score(pred, labels_test)

print 'C Parameter : 1 :  ' ,acc

clf1 = SVC(kernel="rbf" , C= 1000.0)


#### now your job is to fit the classifier
#### using the training features/labels, and to
#### make a set of predictions on the test data
clf1.fit(features_train, labels_train)  
#### store your predictions in a list named pred
pred = clf1.predict( features_test)
acc1 = accuracy_score(pred, labels_test)

print 'C Parameter1000 : ' ,acc1

clf1 = SVC(kernel="rbf" , C= 10000.0)


#### now your job is to fit the classifier
#### using the training features/labels, and to
#### make a set of predictions on the test data
clf1.fit(features_train, labels_train)  
#### store your predictions in a list named pred
pred = clf1.predict( features_test)
acc1 = accuracy_score(pred, labels_test)

print 'C Parameter10000 : ' ,acc1


# ## Optimized RBF vs. Linear SVM: Accuracy
# 
# Now that you’ve optimized C for the RBF kernel, go back to using the full training set. In general, having a larger training set will improve the performance of your algorithm, so (by tuning C and training on a large dataset) we should get a fairly optimized result. What is the accuracy of the optimized SVM?
# 
# 
# 

# In[1]:


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


#features_train = features_train[:len(features_train)/100] 
# labels_train = labels_train[:len(labels_train)/100] 


clf = SVC(kernel="rbf" , C = 10000.0 )


#### now your job is to fit the classifier
#### using the training features/labels, and to
#### make a set of predictions on the test data
clf.fit(features_train, labels_train)  
#### store your predictions in a list named pred
pred = clf.predict( features_test)
acc = accuracy_score(pred, labels_test)

print acc


# ## Extracting Predictions from an SVM
# 
# What class does your SVM (0 or 1, corresponding to Sara and Chris respectively) predict for element 10 of the test set? The 26th? The 50th? (Use the RBF kernel, C=10000, and 1% of the training set. Normally you'd get the best results using the full training set, but we found that using 1% sped up the computation considerably and did not change our results--so feel free to use that shortcut here.)
# And just to be clear, the data point numbers that we give here (10, 26, 50) assume a zero-indexed list. So the correct answer for element #100 would be found using something like answer=predictions[100]

# In[2]:

pred[10] ,pred[26],pred[50]


# ## How Many Chris Emails Predicted?
# There are over 1700 test events--how many are predicted to be in the “Chris” (1) class? (Use the RBF kernel, C=10000., and the full training set.)
# 
# 

# In[3]:

len(pred)


# In[8]:

chris_emails = 0 ;
sara_emails = 0
for item in pred :
    if(item == 1):
        chris_emails =  chris_emails + 1
    else : 
        sara_emails = sara_emails+1
print chris_emails


# In[ ]:



