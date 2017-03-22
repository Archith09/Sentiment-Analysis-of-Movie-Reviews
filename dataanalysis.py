import matplotlib.pyplot as plt
from pre_processing import PreProcess
import numpy as np

preprocess = PreProcess("data/train", "data/test")
preprocess.read_train_test_data()
preprocess.getTfIdf()
target_labels = preprocess.train_target
total_no = target_labels.shape[0]
counts = np.bincount(target_labels)

colors = ['red', 'yellow', 'green']
labels = 'Negative', 'Neutral', 'Positive'
sizes = [counts[0]*100/float(total_no), counts[1]*100/float(total_no), counts[2]*100/float(total_no)]
plt.pie(sizes,labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
# Set aspect ratio to be equal so that pie is drawn as a circle.
plt.axis('equal')

fig = plt.figure()
plt.show()