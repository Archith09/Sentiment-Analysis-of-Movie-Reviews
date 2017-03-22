from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from pre_processing import PreProcess
from sklearn import svm
from sklearn import metrics
from sklearn.cross_validation import cross_val_score

preprocess = PreProcess("data/train", "data/test")
preprocess.read_train_test_data()
preprocess.getTfIdf()

svm_clf = svm.SVC(kernel='linear', probability=True)

scores = cross_val_score(svm_clf, preprocess.traintfIdf, preprocess.train_target, cv=3)
print "the cross validated accuracy on training is " + str(scores)
print("the cross validated accuracy(standard deviation) on training is: %0.4f (+/- %0.4f)" % (
scores.mean(), scores.std() * 2))

svm_clf.fit(preprocess.traintfIdf, preprocess.train_target)
# finding the training and test predictions
train_pred_svm = svm_clf.predict(preprocess.traintfIdf)
test_pred_svm = svm_clf.predict(preprocess.testtfIdf)

svm_train_accuracy = metrics.accuracy_score(preprocess.train_target, train_pred_svm)
svm_test_accuracy = metrics.accuracy_score(preprocess.test_target, test_pred_svm)
svm_train_prec = metrics.precision_score(preprocess.train_target, train_pred_svm, average="macro")
svm_test_prec = metrics.precision_score(preprocess.test_target, test_pred_svm, average="macro")
svm_train_recall = metrics.recall_score(preprocess.train_target, train_pred_svm,average="macro")
svm_test_recall = metrics.recall_score(preprocess.test_target, test_pred_svm, average="macro")

print "Scores\t\t" + "SVM Cosine Kernel Metrics"
print "Train Accuracy" + "\t"  + str(svm_train_accuracy)
print "Test Accuracy" + "\t"  + str(svm_test_accuracy)
print "Train Precision" + "\t" + str(svm_train_prec)
print "Test Precision" + "\t" + str(svm_test_prec)
print "Train Recall" + "\t" + str(svm_train_recall)
print "Test Recall" + "\t\t" + str(svm_test_recall)
