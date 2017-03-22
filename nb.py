from sklearn.grid_search import GridSearchCV
from pre_processing import PreProcess
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.cross_validation import cross_val_score

preprocess = PreProcess("data/train", "data/test")
preprocess.read_train_test_data()
preprocess.getTfIdf()

nb_clf = MultinomialNB(alpha=0.1, fit_prior=True, class_prior=None)

scores = cross_val_score(nb_clf, preprocess.traintfIdf, preprocess.train_target, cv=3)
print("the cross validated accuracy on training is " + str(scores))
print("the cross validated accuracy(standard deviation) on training is: %0.4f (+/- %0.4f)" % (
scores.mean(), scores.std() * 2))

nb_clf.fit(preprocess.traintfIdf, preprocess.train_target)
# finding the training and test predictions
train_pred_nb = nb_clf.predict(preprocess.traintfIdf)
test_pred_nb = nb_clf.predict(preprocess.testtfIdf)

nb_train_accuracy = metrics.accuracy_score(preprocess.train_target, train_pred_nb)
nb_test_accuracy = metrics.accuracy_score(preprocess.test_target, test_pred_nb)
nb_train_prec = metrics.precision_score(preprocess.train_target, train_pred_nb, average="macro")
nb_test_prec = metrics.precision_score(preprocess.test_target, test_pred_nb, average="macro")
nb_train_recall = metrics.recall_score(preprocess.train_target, train_pred_nb,average="macro")
nb_test_recall = metrics.recall_score(preprocess.test_target, test_pred_nb, average="macro")

print("Scores\t\t" + "Multinomial Naive Bayes")
print("Train Accuracy" + "\t"  + str(nb_train_accuracy))
print("Test Accuracy" + "\t"  + str(nb_test_accuracy))
print("Train Precision" + "\t" + str(nb_train_prec))
print("Test Precision" + "\t" + str(nb_test_prec))
print("Train Recall" + "\t" + str(nb_train_recall))
print("Test Recall" + "\t\t" + str(nb_test_recall))