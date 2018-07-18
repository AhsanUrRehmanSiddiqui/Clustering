import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from scipy.stats import mode

digits = load_digits()
digits.data.shape

kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits.data)
print(kmeans.cluster_centers_.shape)

print(clusters.size)

labels = np.zeros_like(clusters)
print(accuracy_score(digits.target, labels))

for i in range(10):
    mask = (clusters == i)
    #print("iteration",i, mask)
    #print("digits,taget[",i,"]", digits.target[mask])
    labels[mask] = mode(digits.target[mask])[0]

print(accuracy_score(digits.target, labels))
