from sklearn.grid_search import GridSearchCV
from pre_processing import PreProcess
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression as LR
# from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn import svm


preprocess = PreProcess("data/train", "data/test")
preprocess.read_train_test_data()
preprocess.getTfIdf()

# softmax_clf = LR(multi_class='ovr')

# softmax_clf = RandomForestClassifier(n_estimators=1500, max_depth=10, min_samples_split=2, random_state=0)

# softmax_clf = ExtraTreesClassifier(n_estimators=1000, max_depth=15, min_samples_split=2, random_state=0)

# softmax_clf = BaggingClassifier(KNeighborsClassifier(), max_samples=0.7, max_features=1)

# softmax_clf = AdaBoostClassifier(n_estimators=100)

# softmax_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5),n_estimators=100, algorithm="SAMME")

# softmax_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1, max_depth=1, random_state=0)

# softmax_clf = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)

# softmax_clf = GradientBoostingClassifier(n_estimators=100, max_depth=1, random_state=0)


# works best till now
softmax_clf = AdaBoostClassifier(MultinomialNB(alpha=0.1,fit_prior=True, class_prior=None),n_estimators=400)

# softmax_clf = AdaBoostClassifier(svm.SVC(kernel='linear', probability=True),n_estimators=5)

#scores = cross_val_score(softmax_clf, preprocess.traintfIdf, preprocess.train_target, cv=3)
#print "the cross validated accuracy on training is " + str(scores)
#print("the cross validated accuracy(standard deviation) on training is: %0.4f (+/- %0.4f)" % (
#scores.mean(), scores.std() * 2))

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




# X, y = make_blobs(n_samples=10000, n_features=10, centers=100,
#     random_state=0)

# clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
# scores = cross_val_score(clf, X, y)
# scores.mean()                             


# clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
# scores = cross_val_score(clf, X, y)
# scores.mean()                             


# clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
# scores = cross_val_score(clf, X, y)
# scores.mean()