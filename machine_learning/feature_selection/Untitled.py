
# coding: utf-8

# ## Feature Selection Mini-Project
# Katie explained in a video a problem that arose in preparing Chris and Sara’s email for the author identification project; it had to do with a feature that was a little too powerful (effectively acting like a signature, which gives an arguably unfair advantage to an algorithm). You’ll work through that discovery process here.
# 

# Q : If a decision tree is overfit, would you expect the accuracy on a test set to be very high or pretty low?
# _**low**_
# 
# Q.If a decision tree is overfit, would you expect high or low accuracy on the training set?
# _**High**_

# Q. How many training points are there, according to the starter code?
# 
# Ans  : 150

# ## What’s the accuracy of the decision tree 

# In[ ]:

import pickle
import numpy
numpy.random.seed(42)
from sklearn import tree
from sklearn.metrics import accuracy_score


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "r"))
authors = pickle.load( open(authors_file, "r") )



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]

clf =  tree.DecisionTreeClassifier()
clf.fit(features_train ,labels_train)
pred =  clf.predict(features_test)

accuracy = accuracy_score(pred , labels_test)

print accuracy


# ## What’s the importance of the most important feature? What is the number of this feature
# 

# In[ ]:

high =  max(clf.feature_importances_)
position = 0
for i in range (0 , len(clf.feature_importances_)) :
    if(clf.feature_importances_[i] ==max(clf.feature_importances_)  ) :
        pos = i 
        
print "Importance of the feature " , high 
print "number of the feature " , pos
 
 


# In[ ]:




# In[ ]:



