from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# The dataset used here is a.csv stored in the anaconda installed folder
data = pd.read_csv("a.csv")
x1 = data['x'].values
x2 = data['y'].values
print(data)
x = np.asarray(list(zip(x1, x2)))

# Visualize the data using a scatter plot
plt.scatter(x1, x2)
plt.show()

markers = ['s', 'o', 'v']
k = 3
clusters = KMeans(n_clusters=k).fit(x)

# Plot the data points with different markers based on their cluster assignments
for i, L in enumerate(clusters.labels_):
    plt.plot(x1[i], x2[i], marker=markers[L])
