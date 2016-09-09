#!/usr/bin/python


"""
    starter code for the evaluation mini-project
    start by copying your trained/tested POI identifier from
    that you built in the validation mini-project

    the second step toward building your POI identifier!

    start by loading/formatting the data

"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

### your code goes here 

from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.cross_validation import train_test_split as tts

features_train, features_test, labels_train,labels_test = tts(features, labels, test_size=0.3, random_state=42)

print 'Baseline accuracy:',list(labels_test).count(0)/float(len(labels_test))
clf = DTC()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

print 'Predicted number of person\'s of interest',list(pred).count(1)
print('Accuracy:',accuracy_score(labels_test,pred))
print('Precision:',precision_score(labels_test,pred))
print('Recall:',recall_score(labels_test,pred))


