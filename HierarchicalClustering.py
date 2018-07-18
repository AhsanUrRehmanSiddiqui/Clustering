import pandas as pd
#%matplotlib inline
import matplotlib
matplotlib.use('TkAgg')
import tkinter
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering

customer_data = pd.read_csv('shopping_data.csv')
print(customer_data.shape)
print(customer_data.head())

data = customer_data.iloc[:, 3:5].values



plt.figure(figsize=(10, 7))
plt.title("Customer Dendograms")
dend = shc.dendrogram(shc.linkage(data, method='ward'))

#plt.show()

cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
print(cluster.fit_predict(data))

plt.figure(figsize=(10, 7))
plt.scatter(data[:,0], data[:,1], c=cluster.labels_, cmap='rainbow')
plt.show()
