from sklearn.metrics import precision_score, recall_score, accuracy_score
import numpy as np
import pickle
import json

# Read in data
X_test = np.genfromtxt("data/test_features.csv")
y_test = np.genfromtxt("data/test_labels.csv")

with open('model.pickle', 'rb') as f:
    clf = pickle.load(f)

print("Testing model...")
y_pred = clf.predict(X_test)
acc = clf.score(X_test, y_test)


with open("metrics.json", 'w') as outfile:
    json.dump(
        {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred)
        }, outfile)
