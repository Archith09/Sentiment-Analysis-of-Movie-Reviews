from sklearn.grid_search import GridSearchCV
from pre_processing import PreProcess
from sklearn.multiclass import OneVsRestClassifier
# from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.cross_validation import cross_val_score

preprocess = PreProcess("data/train", "data/test")
preprocess.read_train_test_data()
preprocess.getTfIdf()

lr_clf = OneVsRestClassifier(LinearSVC(random_state=0))
# lr_clf = OutputCodeClassifier(LinearSVC(random_state=0), code_size=2, random_state=0)


# scores = cross_val_score(lr_clf, preprocess.traintfIdf, preprocess.train_target, cv=3)
# print("the cross validated accuracy on training is " + str(scores))
# print("the cross validated accuracy(standard deviation) on training is: %0.4f (+/- %0.4f)" % (
# scores.mean(), scores.std() * 2))

lr_clf.fit(preprocess.traintfIdf, preprocess.train_target)
# finding the training and test predictions
train_pred_knn = lr_clf.predict(preprocess.traintfIdf)
test_pred_knn = lr_clf.predict(preprocess.testtfIdf)

lr_train_accuracy = metrics.accuracy_score(preprocess.train_target, train_pred_knn)
lr_test_accuracy = metrics.accuracy_score(preprocess.test_target, test_pred_knn)
lr_train_prec = metrics.precision_score(preprocess.train_target, train_pred_knn, average="macro")
lr_test_prec = metrics.precision_score(preprocess.test_target, test_pred_knn, average="macro")
lr_train_recall = metrics.recall_score(preprocess.train_target, train_pred_knn,average="macro")
lr_test_recall = metrics.recall_score(preprocess.test_target, test_pred_knn, average="macro")

print("Scores\t\t" + "Multiclass Learning")
print("Train Accuracy" + "\t"  + str(lr_train_accuracy))
print("Test Accuracy" + "\t"  + str(lr_test_accuracy))
print("Train Precision" + "\t" + str(lr_train_prec))
print("Test Precision" + "\t" + str(lr_test_prec))
print("Train Recall" + "\t" + str(lr_train_recall))
print("Test Recall" + "\t\t" + str(lr_test_recall))