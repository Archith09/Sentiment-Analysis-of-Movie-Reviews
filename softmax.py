from sklearn.grid_search import GridSearchCV
from pre_processing import PreProcess
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression as LR

preprocess = PreProcess("data/train", "data/test")
preprocess.read_train_test_data()
preprocess.getTfIdf()

softmax_clf = LR(multi_class='ovr')

scores = cross_val_score(softmax_clf, preprocess.traintfIdf, preprocess.train_target, cv=3)
print "the cross validated accuracy on training is " + str(scores)
print("the cross validated accuracy(standard deviation) on training is: %0.4f (+/- %0.4f)" % (
scores.mean(), scores.std() * 2))

softmax_clf.fit(preprocess.traintfIdf, preprocess.train_target)

train_pred_softmax = softmax_clf.predict(preprocess.traintfIdf)
test_pred_softmax = softmax_clf.predict(preprocess.testtfIdf)

softmax_train_accuracy = metrics.accuracy_score(preprocess.train_target, train_pred_softmax)
softmax_test_accuracy = metrics.accuracy_score(preprocess.test_target, test_pred_softmax)
softmax_train_prec = metrics.precision_score(preprocess.train_target, train_pred_softmax, average="macro")
softmax_test_prec = metrics.precision_score(preprocess.test_target, test_pred_softmax, average="macro")
softmax_train_recall = metrics.recall_score(preprocess.train_target, train_pred_softmax,average="macro")
softmax_test_recall = metrics.recall_score(preprocess.test_target, test_pred_softmax, average="macro")

print "Scores\t\t" + "softmax Metrics"
print "Train Accuracy" + "\t"  + str(softmax_train_accuracy)
print "Test Accuracy" + "\t"  + str(softmax_test_accuracy)
print "Train Precision" + "\t" + str(softmax_train_prec)
print "Test Precision" + "\t" + str(softmax_test_prec)
print "Train Recall" + "\t" + str(softmax_train_recall)
print "Test Recall" + "\t\t" + str(softmax_test_recall)
