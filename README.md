# Principal Component Analysis (PCA) on Breast Cancer Dataset

## Overview

This project demonstrates the application of Principal Component Analysis (PCA) for dimensionality reduction using the Breast Cancer dataset from Scikit-learn. The goal is to visualize how PCA can transform high-dimensional data into a 2D space while preserving the essential structure and variance.

## Objectives

* Load and explore the Breast Cancer dataset
* Standardize the features using `StandardScaler`
* Apply PCA to reduce the dataset to two principal components
* Visualize the transformed data to observe class separability

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Matplotlib

## Key Steps

1. Load the dataset using `load_breast_cancer()` from Scikit-learn
2. Create a DataFrame using Pandas for easy exploration
3. Standardize the dataset using `StandardScaler` to normalize feature values
4. Apply `PCA(n_components=2)` to project data onto two principal components
5. Visualize the PCA-transformed data with a scatter plot

## Code Snippet

```python
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load dataset
cancer = load_breast_cancer()

# Create DataFrame
df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)

# Standardize data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Apply PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# Plot PCA results
plt.scatter(x=pca_data[:, 0], y=pca_data[:, 1], c=cancer['target'], alpha=0.5)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA Visualization')
plt.show()

# Projection on PC1
plt.scatter(pca_data, [0]*len(pca_data))
plt.title('Projection onto PC1')
plt.show()
```

## Insights

The PCA plot demonstrates how two principal components can capture the majority of the variance from thirty original features. The scatter plot shows that the two classes are reasonably well-separated, highlighting PCA’s ability to simplify complex datasets without significant loss of information.

## Next Steps

* Apply PCA on other real-world datasets
* Compare variance explained by each principal component
* Integrate PCA as a preprocessing step in classification models



**Abdullah Muhammad Khalid**
[Portfolio Website](https://abdullahkhalid.vercel.app/)
[GitHub Profile](https://github.com/abdullahkhvlid)

---

Would you like me to add a **Results section** with a short explanation of what the plots mean (so it looks like a complete ML case study on GitHub)?
