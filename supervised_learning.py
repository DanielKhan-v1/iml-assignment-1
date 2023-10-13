import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from sklearn.datasets import make_classification

# Define the number of samples per class
samples_per_class = [40, 40]  # For a binary classification task

# Calculate the weights based on the desired number of samples per class
total_samples = sum(samples_per_class)
class_weights = [n / total_samples for n in samples_per_class]

X, y = make_classification(
    n_samples=sum(samples_per_class),
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=1,
    n_clusters_per_class=1,
    weights=class_weights
)


count_0 = np.count_nonzero(y == 0)
count_1 = np.count_nonzero(y == 1)

print(f"Count of 0s: {count_0}")
print(f"Count of 1s: {count_1}")

