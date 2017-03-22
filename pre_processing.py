from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import time
from sklearn import svm
from sklearn import metrics
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os

class PreProcess:

    def __init__(self, train_loc, test_loc):
        self.test_loc = test_loc
        self.train_loc = train_loc
        self.training_data = []
        self.train_target = None
        self.test_data = []
        self.test_target = None


    def read_train_test_data(self):
        f_train = open(os.path.join(self.train_loc, 'reviews.txt'), 'r')
        f_test = open(os.path.join(self.test_loc, 'reviews.txt'), 'r')
        for line in f_train:
            self.training_data.append(line)
        for line in f_test:
            self.test_data.append(line)
        f_train = open(os.path.join(self.train_loc, 'labels.txt'), 'r')
        f_test = open(os.path.join(self.test_loc, 'labels.txt'), 'r')
        temp_labels_test = []
        temp_labels_train = []
        for line in f_train:
            temp_labels_train.append(int(line))
        for line in f_test:
            temp_labels_test.append(int(line))
        self.train_target = np.array(temp_labels_train)
        self.test_target = np.array(temp_labels_test)


    def getTfIdf(self):
        count_vect = CountVectorizer(stop_words='english', max_features=4000)
        X_train_fit = count_vect.fit(self.training_data)
        X_train_counts = X_train_fit.transform(self.training_data)
        tfIdfFit = TfidfTransformer(use_idf=True, norm='l2', sublinear_tf=True).fit(X_train_counts)
        self.traintfIdf = tfIdfFit.transform(X_train_counts)
        X_test_counts = X_train_fit.transform(self.test_data)
        self.testtfIdf = tfIdfFit.transform(X_test_counts)


if __name__=="__main__":
    preprocess = PreProcess("data/train", "data/test")
    preprocess.read_train_test_data()
    preprocess.getTfIdf()
    preprocess.training_data.extend(preprocess.test_data)
    combined_labels = np.concatenate((preprocess.train_target, preprocess.test_target), axis=0)
    combined_data = np.array(preprocess.training_data)
    idx = np.arange(np.size(combined_data))
    np.random.seed(13)
    np.random.shuffle(idx)
    combined_data = combined_data[idx]
    combined_labels = combined_labels[idx]
    combined_labels = combined_labels.astype(int)
    np.savetxt("train_data.txt", combined_data[:4506], fmt="%s", newline="")
    np.savetxt("test_data.txt", combined_data[4506:], fmt="%s", newline="")
    np.savetxt("train_label.txt", combined_labels[:4506], fmt="%d")
    np.savetxt("test_label.txt", combined_labels[4506:], fmt="%d")
