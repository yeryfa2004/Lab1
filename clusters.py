import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth

ms_model = None

def trainModel(data):
    bandwidth = estimate_bandwidth(data, quantile=0.15, n_samples=len(data))
    global ms_model
    ms_model = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms_model.fit(data)

def getClusters():
    cluster_centers = ms_model.cluster_centers_
    labels = ms_model.labels_
    num_clusters = len(np.unique(labels))
    return labels, cluster_centers, num_clusters

def getClusterAreas(data):
    step_size = 0.01
    x_min, x_max = data[:,0].min() - 1, data[:,0].max() + 1
    y_min, y_max = data[:,1].min() - 1, data[:,1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    output = ms_model.predict(grid_points)
    output = output.reshape(xx.shape)
    return output, xx, yy
