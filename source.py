from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def kmeans(img_1d, k_clusters, max_iter, init_centroids="random"):
    def initialize_centroids(img_1d, k_clusters, init_centroids):
        if init_centroids == "random":
            return np.random.rand(k_clusters, img_1d.shape[1]) * 255
        elif init_centroids == "in_pixels":
            indices = np.random.randint(0, len(img_1d), k_clusters)
            return img_1d[indices]

    centroids = initialize_centroids(img_1d, k_clusters, init_centroids)
    for _ in range(max_iter):
        distances = np.sqrt(((img_1d - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        centroids = np.array(
            [img_1d[labels == k].mean(axis=0) for k in range(k_clusters)]
        )

    return centroids, labels


print(
    "Run the program again if it occur RuntimeWarning: Mean of empty slice or invalid value encountered"
)


# Load the image using PIL
# img = Image.open(r"C:\Users\pkhoa\Downloads\another adachi.jpg")
img = input("Enter the path of the image: ")

img = Image.open(img)
img_copy = img.copy()
# Convert to numpy array and reshape to 1D
img_np = np.array(img)
img_1d = img_np.reshape(-1, 3)

# Define the number of clusters
k_clusters = 3

np.seterr(divide="ignore", invalid="ignore")

# Run the K-Means algorithm
centroids, labels = kmeans(
    img_1d, k_clusters=k_clusters, max_iter=100, init_centroids="random"
)

# Reshape labels back to 2D
labels_2d = labels.reshape(img_np.shape[0], img_np.shape[1])

# Create a new image with the labels, which represent the colors
new_img = np.zeros(img_np.shape, dtype=np.uint8)
for i in range(k_clusters):
    new_img[labels_2d == i] = centroids[i]


# Display the image using matplotlib

plt.imshow(new_img)
plt.title("Quantized Image")


plt.tight_layout()
plt.show()
