# 🛍️ Mall Customer Segmentation using K-Means Clustering

## 📌 Project Overview

This project performs customer segmentation using the K-Means Clustering algorithm. Customers are grouped based on their Annual Income and Spending Score to identify different purchasing behaviors.

The project includes:

- Data Loading
- Data Exploration
- Feature Selection
- Feature Scaling
- Elbow Method
- Optimal K Selection
- K-Means Clustering
- Cluster Visualization
- Cluster Analysis

---

## 📂 Project Structure

```text
Mall-Customer-Segmentation/
│
├── Mall_Customer_Segmentation.ipynb
├── Mall_Customers.csv
├── Mall_Customers_Clustered.csv
├── Elbow_Method.png
├── KMeans_Clusters.png
├── Workflow.md
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 📊 Dataset

The dataset contains customer information including:

- CustomerID
- Gender
- Age
- Annual Income (k$)
- Spending Score (1–100)

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook
- Git & GitHub

---

## 📈 Project Workflow

### 1. Import Libraries

Imported all required Python libraries.

---

### 2. Load Dataset

Loaded the Mall Customers dataset and checked its structure.

---

### 3. Data Exploration

Performed:

- Dataset overview
- Data types
- Summary statistics

---

### 4. Feature Selection

Selected:

- Annual Income (k$)
- Spending Score (1–100)

---

### 5. Feature Scaling

Applied StandardScaler to normalize the selected features.

---

### 6. Elbow Method

Calculated WCSS for different values of K to determine the optimal number of clusters.

---

### 7. K-Means Clustering

Trained the K-Means model using the optimal number of clusters.

---

### 8. Cluster Visualization

Visualized customer clusters and cluster centroids.

---

### 9. Save Results

Saved the clustered dataset for future analysis.

---

## 📌 Machine Learning Concepts Covered

- Unsupervised Learning
- Clustering
- K-Means Algorithm
- Feature Scaling
- Elbow Method
- Centroids
- Cluster Analysis

---

## 🚀 Future Improvements

- Silhouette Score Analysis
- Hierarchical Clustering
- DBSCAN Clustering
- PCA for Visualization
- Interactive Dashboard using Streamlit

---

## 👩‍💻 Author

**Geeta Kadam**

GitHub: https://github.com/geetakadam275