from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import BaggingClassifier
import numpy as np
import pandas

# Load data
d = pandas.read_csv('train.csv')
y = np.array(d.target)  # Labels
X = np.array(d.iloc[:,2:])  # Features

#svm.decision function
# Split into train/test folds
# TODO
np.random.seed(0)
s = np.arange(X.shape[0])
np.random.shuffle(s)
X = X[s]
y = y[s]
xtr = X[:100000, :]
ytr = y[:100000,]
xte = X[100000:, :]
yte = y[100000:,]
clf_train = LinearSVC(dual=False)
clf_train.fit(xtr, ytr)
yhat = clf_train.decision_function(xte)
auc1 = roc_auc_score(yte, yhat)
print('Linear Kernel SVM auc', auc1)

predictions = []
j = 0
#bagging
for i in range(0, xtr.shape[0], 10000):
    subset_train = xtr[i:(i + 10000), :]
    subset_labels = ytr[i:(i + 10000), ]
    subset_test = (xte[i:(i + 10000), :])
    subset_test_labels = (yte[i:(i + 10000), ])
    clf_train = SVC(kernel='poly', degree=3, gamma='auto')
    clf_train.fit(subset_train, subset_labels)
    predictions.append(clf_train.decision_function(xte))
    j += 1

predictions = np.asarray(predictions)
predictions = np.sum(predictions, axis=0)
predictions /= 10
auc2 = roc_auc_score(yte, predictions)
print('Polynomial Kernel SVM auc', auc2)
# print(len(subset_train))


# Linear SVM
# TODO

# Non-linear SVM (polynomial kernel)
# TODO

# Apply the SVMs to the test set
#yhat1 = ...  # Linear kernel
#yhat2 = ...  # Non-linear kernel

# Compute AUC
#auc1 = ...
#auc2 = ...

# print(auc1)
# print(auc2)
