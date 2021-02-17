from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Read in data
X_test = np.genfromtxt("data/test_features.csv")
y_test = np.genfromtxt("data/test_labels.csv")

with open('model.pickle', 'rb') as f:
    clf = pickle.load(f)

print("Testing model...")
acc = clf.score(X_test, y_test)
