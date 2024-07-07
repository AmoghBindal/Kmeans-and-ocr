import numpy as np 
import matplotlib.pyplot as plt

class kmeansclustering:
    def __init__(self, k = 100):
        self.k = k
        self.centroids = None

    def vec_dist (data_point, centroids):
        return np.sqrt(np.sum((data_point-centroids)**2, axis= 1))

    def fit(self, X, maxiterations=200):
        self.centroids = X[np.random.randint(X.shape[0], size=self.k)]

        for _ in range(maxiterations):
            y=[]

            for data_point in X:
                distances = kmeansclustering.vec_dist(data_point, self.centroids)
                cluster_num = np.argmin(distances)
                y.append(cluster_num)

            y = np.array(y)

            cluster_indices = []

            for i in range (self.k):
                cluster_indices.append(np.argwhere(y ==i))

            cluster_centers =[]

            for i, indices in enumerate(cluster_indices):
                if len(indices) == 0:
                    cluster_centers.append(self.centroids[i])

                else:
                    cluster_centers.append(np.mean(X[indices], axis= 0)[0])

            if np.max(self.centroids - np.array(cluster_centers)) < 0.0001:
                break
            else:
                self.centroids = np.array(cluster_centers)

        return y
    


random_points = np.random.randint(0,100, (100,2))

kmeams = kmeansclustering(k=3)

labels = kmeams.fit(random_points)

plt.scatter(random_points[:,0],random_points[:,1], c=labels, s=10)
plt.scatter(kmeams.centroids[:, 0], kmeams.centroids[:,1], c=range(len(kmeams.centroids)), marker='*')

plt.show()