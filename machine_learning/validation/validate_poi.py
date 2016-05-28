#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn import cross_validation
from sklearn import tree
from sklearn.metrics import accuracy_score

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

#features_train , features_test , labels_train , labels_test = cross_validation.train_test_split(data , labels , test_size =.4 , random_state =0 )
clf = tree.DecisionTreeClassifier()
clf.fit(features , labels)
pred = clf.predict(features)

acc = accuracy_score(pred , labels )

print "accuracy is "  ,acc ;
features_train , features_test , labels_train , labels_test = cross_validation.train_test_split(features , labels , test_size =.3 , random_state =42 )
clf = tree.DecisionTreeClassifier()
clf.fit(features_train , labels_train)
pred = clf.predict(features_test)

acc2 = accuracy_score(pred , labels_test )

print "updated accuracy is " , acc2




### it's all yours from here forward!  


