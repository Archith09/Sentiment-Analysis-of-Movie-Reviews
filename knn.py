from sklearn.grid_search import GridSearchCV
from pre_processing import PreProcess
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.cross_validation import cross_val_score

preprocess = PreProcess("data/train", "data/test")
preprocess.read_train_test_data()
preprocess.getTfIdf()

knn_clf = KNeighborsClassifier(n_neighbors=5, algorithm='auto')
# (n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1, **kwargs)

scores = cross_val_score(knn_clf, preprocess.traintfIdf, preprocess.train_target, cv=3)
print("the cross validated accuracy on training is " + str(scores))
print("the cross validated accuracy(standard deviation) on training is: %0.4f (+/- %0.4f)" % (
scores.mean(), scores.std() * 2))

knn_clf.fit(preprocess.traintfIdf, preprocess.train_target)
# finding the training and test predictions
train_pred_knn = knn_clf.predict(preprocess.traintfIdf)
test_pred_knn = knn_clf.predict(preprocess.testtfIdf)

knn_train_accuracy = metrics.accuracy_score(preprocess.train_target, train_pred_knn)
knn_test_accuracy = metrics.accuracy_score(preprocess.test_target, test_pred_knn)
knn_train_prec = metrics.precision_score(preprocess.train_target, train_pred_knn, average="macro")
knn_test_prec = metrics.precision_score(preprocess.test_target, test_pred_knn, average="macro")
knn_train_recall = metrics.recall_score(preprocess.train_target, train_pred_knn,average="macro")
knn_test_recall = metrics.recall_score(preprocess.test_target, test_pred_knn, average="macro")

print("Scores\t\t" + "kNearestNeighbors")
print("Train Accuracy" + "\t"  + str(knn_train_accuracy))
print("Test Accuracy" + "\t"  + str(knn_test_accuracy))
print("Train Precision" + "\t" + str(knn_train_prec))
print("Test Precision" + "\t" + str(knn_test_prec))
print("Train Recall" + "\t" + str(knn_train_recall))
print("Test Recall" + "\t\t" + str(knn_test_recall))