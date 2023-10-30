import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import tree
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

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

# Separate data points for each class
X_class0 = X[y == 0]
X_class1 = X[y == 1]

# Create a scatter plot to visualize the data
plt.scatter(X_class0[:, 0], X_class0[:, 1], c='blue', label='Class 0')
plt.scatter(X_class1[:, 0], X_class1[:, 1], c='red', label='Class 1')

# Add labels and legend
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

# Show the plot
plt.title('Synthetic Classification Data')
plt.show()

count_0 = np.count_nonzero(y == 0)
count_1 = np.count_nonzero(y == 1)

print(f"Count of 0s: {count_0}")
print(f"Count of 1s: {count_1}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the number of outliers to add for each class
num_outliers = 4

# Create outliers for Class 0
outliers_class0 = np.random.rand(num_outliers, X.shape[1]) * 10
# Assign label 0 to the outliers for Class 0
outliers_class0_labels = np.zeros(num_outliers)

# Create outliers for Class 1
outliers_class1 = np.random.rand(num_outliers, X.shape[1]) * 10
# Assign label 1 to the outliers for Class 1
outliers_class1_labels = np.ones(num_outliers)

# Combine Dataset 1 and outliers with explicit labels to create Dataset 2
X_dataset2 = np.vstack((X, outliers_class0, outliers_class1))
y_dataset2 = np.hstack((y, outliers_class0_labels, outliers_class1_labels))

X_train2, X_test2, y_train2, y_test2 = train_test_split(X_dataset2, y_dataset2, test_size=0.2, random_state=42)

print(X_train2.shape, X_test2.shape, X_train.shape, X_test.shape)

# ---------------------------------------------------------------
# DECISION TREE
# ---------------------------------------------------------------

print("\n---------------------------------------------------------------")
print("DECISION TREE")
print("---------------------------------------------------------------")

# Create and train a Decision Tree classifier
clf_tree = tree.DecisionTreeClassifier()
clf_tree = clf_tree.fit(X_train, y_train)

# Plot Decision Tree
tree.plot_tree(clf_tree)
plt.show()

# Train a new Decision Tree classifier on Dataset 2
clf_tree = tree.DecisionTreeClassifier()
clf_tree = clf_tree.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf_tree.predict(X_test)

# Calculate accuracy for Dataset 1
accuracy1_train = accuracy_score(y_train, clf_tree.predict(X_train))
accuracy1_test = accuracy_score(y_test, y_pred)

# Report the accuracy for Dataset 1
print("Accuracy on Training Data (Dataset 1): {:.2f}%".format(accuracy1_train * 100))
print("Accuracy on Testing Data (Dataset 1): {:.2f}%".format(accuracy1_test * 100))

# Generate a classification report
report = classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1'])
print("Classification Report for Dataset 1: \n" + report)

# Plot the confusion matrix
conf_matrix_tree = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_tree, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Decision Tree Confusion Matrix (Dataset 1)')
plt.show()

# Create and train a Decision Tree classifier on Dataset 2
clf_tree = clf_tree.fit(X_train2, y_train2)

# Make predictions on the test set for Dataset 2
y_pred2 = clf_tree.predict(X_test2)

# Calculate accuracy for Dataset 2
accuracy2_train = accuracy_score(y_train2, clf_tree.predict(X_train2))
accuracy2_test = accuracy_score(y_test2, y_pred2)

# Report the accuracy for Dataset 2
print("Accuracy on Training Data (Dataset 2): {:.2f}%".format(accuracy2_train * 100))
print("Accuracy on Testing Data (Dataset 2): {:.2f}%".format(accuracy2_test * 100))

# Generate a classification report for Dataset 2
report_dataset2 = classification_report(y_test2, y_pred2, target_names=['Class 0', 'Class 1'])
print("Classification Report after including Dataset 2: \n" + report_dataset2)

# Plot the confusion matrix for Dataset 2
conf_matrix_tree_dataset2 = confusion_matrix(y_test2, y_pred2)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_tree_dataset2, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Decision Tree Confusion Matrix (Dataset 1 and 2)')
plt.show()

# ---------------------------------------------------------------
# K-Nearest Neighbors
# ---------------------------------------------------------------

print("\n---------------------------------------------------------------")
print("K-NEAREST NEIGHBORS")
print("---------------------------------------------------------------")

# clf_knn.fit(X_train2, y_train2)
# Create and train k-NN classifier on Dataset 1
clf_knn = KNeighborsClassifier(n_neighbors=3)
clf_knn = clf_knn.fit(X_train, y_train)

# Make predictions on the test set for Dataset 1
y_pred1 = clf_knn.predict(X_test)

# Calculate accuracy for Dataset 1 on training and testing data
accuracy_train1 = accuracy_score(y_train, clf_knn.predict(X_train))
accuracy_test1 = accuracy_score(y_test, y_pred1)
print("Accuracy on Training Data (Dataset 1): {:.2f}%".format(accuracy_train1 * 100))
print("Accuracy on Testing Data (Dataset 1): {:.2f}%".format(accuracy_test1 * 100))

# Generate a classification report
report = classification_report(y_test, y_pred1, target_names=['Class 0', 'Class 1'])
print("Classification Report for Dataset 1: \n" + report)

# Generate a confusion matrix for Dataset 1
conf_matrix1 = confusion_matrix(y_test, y_pred1)

# Print the confusion matrix for Dataset 1
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix1, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('K-Nearest Neighbors Confusion Matrix (Dataset 1)')
plt.show()

# Train k-NN classifier on Dataset 2
clf_knn = clf_knn.fit(X_train2, y_train2)

# Make predictions on the test set for Dataset 2
y_pred2 = clf_knn.predict(X_test2)

# Calculate accuracy for Dataset 2 on training and testing data
accuracy_train2 = accuracy_score(y_train2, clf_knn.predict(X_train2))
accuracy_test2 = accuracy_score(y_test2, y_pred2)
print("Accuracy on Training Data (Dataset 2): {:.2f}%".format(accuracy_train2 * 100))
print("Accuracy on Testing Data (Dataset 2): {:.2f}%".format(accuracy_test2 * 100))

# Generate a classification report for Dataset 2
report_dataset2 = classification_report(y_test2, y_pred2, target_names=['Class 0', 'Class 1'])
print("Classification Report after including Dataset 2: \n" + report_dataset2)

conf_matrix2 = confusion_matrix(y_test2, y_pred2)

# Plot the confusion matrix for Dataset 2
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix2, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('K-Nearest Neighbors Confusion Matrix (Dataset 1 and 2)')
plt.show()

# ---------------------------------------------------------------
# Naive Bayes
# ---------------------------------------------------------------

print("\n---------------------------------------------------------------")
print("NAIVE BAYES")
print("---------------------------------------------------------------")

clf_NB = GaussianNB()
clf_NB.fit(X_train, y_train)

y_pred = clf_NB.predict(X_test)

# Calculate accuracy for Dataset 1 on training and testing data
accuracy_train1 = accuracy_score(y_train, clf_NB.predict(X_train))
accuracy_test1 = accuracy_score(y_test, y_pred)
print("Accuracy on Training Data (Dataset 1): {:.2f}%".format(accuracy_train1 * 100))
print("Accuracy on Testing Data (Dataset 1): {:.2f}%".format(accuracy_test1 * 100))

# Generate a classification report
report = classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1'])
print("Classification Report for Dataset 1: \n" + report)

conf_matrix_tree = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_tree, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Naive Bayes Confusion Matrix (Dataset 1)')
plt.show()

print(" ")

clf_NB.fit(X_train2, y_train2)

y_pred = clf_NB.predict(X_test2)

# Calculate accuracy for Dataset 2 on training and testing data
accuracy_train2 = accuracy_score(y_train2, clf_NB.predict(X_train2))
accuracy_test2 = accuracy_score(y_test2, y_pred)
print("Accuracy on Training Data (Dataset 2): {:.2f}%".format(accuracy_train2 * 100))
print("Accuracy on Testing Data (Dataset 2): {:.2f}%".format(accuracy_test2 * 100))

# Generate a classification report for Dataset 2
report_dataset2 = classification_report(y_test2, y_pred, target_names=['Class 0', 'Class 1'])
print("Classification Report after including Dataset 2: \n" + report_dataset2)

# Plot confusion matrix
conf_matrix_tree_dataset2 = confusion_matrix(y_test2, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_tree_dataset2, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Naive Bayes Confusion Matrix (Dataset 1 and 2)')
plt.show()

# ---------------------------------------------------------------
# Randon Forest
# ---------------------------------------------------------------

print("\n---------------------------------------------------------------")
print("RANDOM FOREST")
print("---------------------------------------------------------------")

# Create a RandomForestClassifier instance
clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
# Fit the classifier to training data
clf_rf.fit(X_train, y_train)

# Make predictions on test data
y_pred = clf_rf.predict(X_test)

# Calculate accuracy for Dataset 1 on training and testing data
accuracy_train1 = accuracy_score(y_train, clf_rf.predict(X_train))
accuracy_test1 = accuracy_score(y_test, y_pred)
print('Accuracy on Training Data (Dataset 1): {:.2f}%'.format(accuracy_train1 * 100))
print("Accuracy on Testing Data (Dataset 1): {:.2f}%".format(accuracy_test1 * 100))

# Generate a classification report
report = classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1'])
print("Classification Report for Dataset 1: \n" + report)

# Plot confusion matrix
conf_matrix_rf = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Random Forest Confusion Matrix (Dataset 1)')
plt.show()

# Training on Dataset 2
clf_rf.fit(X_train2, y_train2)
# Making predictions on test data
y_pred = clf_rf.predict(X_test2)

# Calculating accuracy for training and testing
accuracy_train2 = accuracy_score(y_train2, clf_rf.predict(X_train2))
accuracy_test2 = accuracy_score(y_test2, y_pred)
print("Accuracy on Training Data (Dataset 2): {:.2f}%".format(accuracy_train2 * 100))
print("Accuracy on Testing Data (Dataset 2): {:.2f}%".format(accuracy_test2 * 100))

# Generating a classification report for Dataset 2
report_dataset2 = classification_report(y_test2, y_pred, target_names=['Class 0', 'Class 1'])
print("Classification Report after including Dataset 2: \n" + report_dataset2)

# Plotting confusion matrix
conf_matrix_rf2 = confusion_matrix(y_test2, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_rf2, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Random Forest Confusion Matrix (Dataset 1 & 2)')
plt.show()

# Create a grid of points to make predictions on
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# Define model names
model_names = ["Decision Tree", "K-Nearest Neighbors", "Naive Bayes", "Random Forest"]

# Create a figure and a set of subplots
plt.figure(figsize=(16, 12))

# Plot the decision boundaries for each model
for i, clf in enumerate([clf_tree, clf_knn, clf_NB, clf_rf]):
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    # Predict the class for each point in the grid
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundaries
    plt.contourf(xx, yy, Z, alpha=0.4)

    # Plot the data points
    plt.scatter(X_class0[:, 0], X_class0[:, 1], c='blue', label='Class 0')
    plt.scatter(X_class1[:, 0], X_class1[:, 1], c='red', label='Class 1')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.title(model_names[i])  # Add model name as the title

# Show the plot
plt.show()
