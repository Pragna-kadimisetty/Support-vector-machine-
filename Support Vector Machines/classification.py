# Support Vector Machine (SVM) for Breast Cancer Dataset
# Linear and Non-linear classification with visualization & image saving

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
import os
import seaborn as sns

# 1. Load dataset
df = pd.read_csv("breast-cancer.csv")  # Change path if needed

# Create output folder for images
os.makedirs("svm_outputs", exist_ok=True)

# Assuming 'diagnosis' is the target
X = df.drop(columns=['diagnosis'])
y = df['diagnosis'].map({'M': 1, 'B': 0})

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Function to plot and save confusion matrix
def save_confusion_matrix(y_true, y_pred, filename, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.savefig(f"svm_outputs/{filename}", dpi=300)
    plt.close()

# 3. Linear SVM
linear_svm = SVC(kernel='linear', C=1)
linear_svm.fit(X_train, y_train)
y_pred_linear = linear_svm.predict(X_test)

# Save report
with open("svm_outputs/linear_svm_report.txt", "w") as f:
    f.write("Linear SVM Results:\n")
    f.write(f"Accuracy: {accuracy_score(y_test, y_pred_linear):.4f}\n")
    f.write(classification_report(y_test, y_pred_linear))

# Save confusion matrix image
save_confusion_matrix(y_test, y_pred_linear, "linear_svm_confusion_matrix.png", "Linear SVM Confusion Matrix")

# 4. Non-linear SVM (RBF Kernel) with Hyperparameter tuning
param_grid = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1]}
rbf_svm = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
rbf_svm.fit(X_train, y_train)
y_pred_rbf = rbf_svm.predict(X_test)

with open("svm_outputs/rbf_svm_report.txt", "w") as f:
    f.write(f"Best Parameters (RBF): {rbf_svm.best_params_}\n")
    f.write(f"Accuracy (RBF): {accuracy_score(y_test, y_pred_rbf):.4f}\n")
    f.write(classification_report(y_test, y_pred_rbf))

# Save confusion matrix image
save_confusion_matrix(y_test, y_pred_rbf, "rbf_svm_confusion_matrix.png", "RBF SVM Confusion Matrix")

# 5. Cross-validation score for RBF
cv_score = cross_val_score(SVC(kernel='rbf', C=rbf_svm.best_params_['C'], gamma=rbf_svm.best_params_['gamma']), X, y, cv=5)
with open("svm_outputs/cv_scores.txt", "w") as f:
    f.write("Cross-Validation Scores:\n")
    f.write(str(cv_score) + "\n")
    f.write(f"Mean CV Accuracy: {np.mean(cv_score):.4f}\n")

# 6. Visualization using PCA (2D projection)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

best_svm = SVC(kernel='rbf', C=rbf_svm.best_params_['C'], gamma=rbf_svm.best_params_['gamma'])
best_svm.fit(X_train_pca, y_train)

# Plot decision boundary & save image
def plot_decision_boundary(model, X, y, filename):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8,6))
    plt.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('SVM Decision Boundary (PCA Projection)')
    plt.savefig(f"svm_outputs/{filename}", dpi=300)
    plt.close()

plot_decision_boundary(best_svm, X_train_pca, y_train, "svm_decision_boundary.png")

print("✅ All reports, confusion matrices, and plots saved in 'svm_outputs' folder.")
