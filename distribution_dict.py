import matplotlib.pyplot as plt
from pre_processing import PreProcess
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from matplotlib_venn import venn3

preprocess = PreProcess("data/train", "data/test")
preprocess.read_train_test_data()
preprocess.getTfIdf()
target_labels = preprocess.train_target
positive_ind = target_labels == 2
negative_ind = target_labels == 0
neutral_ind = target_labels == 1

train_np = np.array(preprocess.training_data)

positive_examples = train_np[positive_ind]
negative_examples = train_np[negative_ind]
neutral_examples = train_np[neutral_ind]

count_vect = CountVectorizer(stop_words='english', max_features=1000)
X_train_fitp = count_vect.fit(positive_examples)

count_vect1 = CountVectorizer(stop_words='english', max_features=1000)
X_train_fitneg = count_vect1.fit(negative_examples)

count_vect2 = CountVectorizer(stop_words='english', max_features=1000)
X_train_fitneut = count_vect2.fit(neutral_examples)

positive = set(X_train_fitp.vocabulary_.keys())
negative = set(X_train_fitneg.vocabulary_.keys())
neutral = set(X_train_fitneut.vocabulary_.keys())

venn3([positive, negative, neutral], ('positive', 'negative', 'neutral'))
plt.show()
