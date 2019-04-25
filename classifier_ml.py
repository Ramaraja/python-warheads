from __future__ import division
from sklearn import datasets
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score

from tabulate import tabulate

wine = datasets.load_wine()


features = wine.data
labels = wine.target

print "Class",labels

print "No of records:", len(features)

print tabulate(features, headers=wine.feature_names)

# print tts(features, labels, test_size=0.2)
# split the data into training and testing
train_feats, test_feats, train_labels, test_labels = tts(features, labels, test_size=0.2)

# SVM with RBF kernel. Default setting of SVM.
clf = svm.SVC()

# clf = svm.SVC(kernel='linear')

# clf = tree.DecisionTreeClassifier()

clf = RandomForestClassifier()

# print the details of the Classifier used
print "Using", clf

# training
clf.fit(train_feats, train_labels)

# predictions
predictions = clf.predict(test_feats)

# predictions = clf.predict([[14,1,3,45,67,34,89,23,7,8,1,1,1]])
print "\nPredictions:", predictions

score = 0
for i in range(len(predictions)):
    if predictions[i] == test_labels[i]:
        score += 1
print "Accuracy:", (score / len(predictions)) * 100, "%"


print accuracy_score(test_labels, predictions)