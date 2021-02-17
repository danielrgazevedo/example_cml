from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pickle


# Read in data
X_train = np.genfromtxt("data/train_features.csv")
y_train = np.genfromtxt("data/train_labels.csv")


# Fit a model
clf = RandomForestClassifier()
# clf = SVC()
clf.fit(X_train, y_train)
print("Training model...")
with open('model.pickle', 'wb') as f:
    pickle.dump(clf, f)


acc = clf.score(X_train, y_train)
with open("acc_train.txt", 'w') as outfile:
    outfile.write("Training Accuracy: " + str(acc) + "\n")


# Plot it
disp = plot_confusion_matrix(clf,
                             X_train,
                             y_train,
                             normalize='true',
                             cmap=plt.cm.Blues)
plt.savefig('confusion_matrix.png')
