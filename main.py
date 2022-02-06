
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X= 1 * np.random.rand(100,2)
X1 = 2 + 2 * np.random.rand(50,2)
X[50:100, :] = X1
plt.scatter(X[ : , 0], X[ :, 1], s = 50, c = 'b')
plt.show()
from sklearn.cluster import KMeans
Kmean = KMeans(n_clusters=2)
Kmean.fit(X)
print(Kmean.cluster_centers_)
plt.scatter(X[ : , 0], X[ : , 1], s =50, c='b')
plt.scatter(0.50782468, 0.46122215, s=200, c='g', marker='s')
plt.scatter(2.99496123, 3.08102225, s=200, c='r', marker='s')
plt.show()
print(Kmean.labels_)
sample_test=np.array([-3.0,-3.0])
second_test=sample_test.reshape(1, -1)
Kmean.predict(second_test)
