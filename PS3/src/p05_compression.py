from matplotlib.image import imread
import matplotlib
matplotlib.use('TkAgg')  # 或 'Qt5Agg'
import numpy as np
import matplotlib.pyplot as plt
def kmeans_compress(img,k=16,max_iter=30):
    h, w, c = img.shape
    X=img.reshape(-1, 3)
    m = X.shape[0]
    random_indices = np.random.choice(m, k, replace=False)
    centroids = X[random_indices]
    for iteration in range(max_iter):
        distances = np.sum((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.zeros((k, 3))
        for j in range(k):
            cluster_points = X[labels == j]
            if len(cluster_points) > 0:
                new_centroids[j, :] = np.mean(cluster_points, axis=0)
            else:
                new_centroids[j, :] = X[np.random.choice(m)]
        if np.allclose(centroids, new_centroids):
            print(f"Converged at iteration {iteration}")
            break
        centroids = new_centroids
    distances = np.sum((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
    labels = np.argmin(distances, axis=1)
    compressed = centroids[labels].reshape(h, w, c)

    return compressed, centroids
small_img = imread('../data/peppers-small.tiff')
compressed_small, centroids = kmeans_compress(small_img, k=16)
# 压缩大图像
large_img = imread('../data/peppers-large.tiff')
h, w, c = large_img.shape
X_large = large_img.reshape(-1, c)
distances = np.sum((X_large[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
labels = np.argmin(distances, axis=1)
compressed_large = centroids[labels].reshape(h, w, c)
print(f"small_img range: [{small_img.min():.4f}, {small_img.max():.4f}]")
print(f"large_img range: [{large_img.min():.4f}, {large_img.max():.4f}]")
# 显示结果
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)

plt.imshow(large_img)
plt.title('Original')
plt.subplot(1, 2, 2)
plt.imshow(compressed_large/255.0)
plt.title('Compressed (16 colors)')
plt.show(block=True)