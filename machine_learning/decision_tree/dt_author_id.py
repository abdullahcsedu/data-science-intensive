
# coding: utf-8

# ## Decision Tree Mini-Project
# 
# In this project, we will again try to identify the authors in a body of emails, this time using a decision tree. The starter code is in decision_tree/dt_author_id.py.

# In[4]:

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.tree  import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


features_train = features_train[:len(features_train)] 
labels_train = labels_train[:len(labels_train)] 

clf = DecisionTreeClassifier(min_samples_split=40)

# train the data set
clf.fit(features_train , labels_train)

pred = clf.predict(features_test)


acc =  accuracy_score(pred , labels_test)

print acc




# ## Speeding Up Via Feature Selection 1
# 
# You found in the SVM mini-project that the parameter tune can significantly speed up the training time of a machine learning algorithm. A general rule is that the parameters can tune the complexity of the algorithm, with more complex algorithms generally running more slowly.
# Another way to control the complexity of an algorithm is via the number of features that you use in training/testing. The more features the algorithm has available, the more potential there is for a complex fit. We will explore this in detail in the “Feature Selection” lesson, but you’ll get a sneak preview now.
# What's the number of features in your data? (Hint: the data is organized into a numpy array where the number of rows is the number of data points and the number of columns is the number of features; so to extract this number, use a line of code like len(features_train[0]).)

# In[11]:

len(features_train[1])


# ## Changing the Number of Features
# 0.967007963595

# In[3]:

len(features_train[1])


# ## Accuracy Using 1% of Features
