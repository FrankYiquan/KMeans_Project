# KMeans Clustering and Evaluation

## Overview

This project implements a custom KMeans clustering algorithm (`MyKMeans`) and evaluates it using the **elbow method**, **Silhouette Score**, and **Normalized Mutual Information (NMI)** score. The project compares the performance of **KMeans** and **KMeans++** on two datasets: **Iris** and **Wine**. The results are used to determine the optimal number of clusters (`k`) for each dataset.

## Files Description

### `main.py`

- **Implemented Methods**:
  - `__init__(k, x, random_state: int, true_label, max_iter=100, min_check=1e-4)`: Initializes the KMeans algorithm with parameters like number of clusters `k`, features `x`, random seed `random_state`, true labels, maximum iterations, and convergence check.
  - `train(method="KMeans")`: Trains the clustering model using either **KMeans** or **KMeans++** method. The default method is **KMeans**.
  - `silhouette_score()`: Returns the Silhouette Score for the clustering result.
  - `normalized_mutual_information_score()`: Returns the Normalized Mutual Information (NMI) score.

### `part_C1.py`

- **Purpose**: Reproduces the table and graph for determining the optimal `k` for the **Iris dataset**.
- **Outputs**:
  - **Graph**: Elbow method plot and Silhouette Score for different values of `k`.
  - **Table**: For each `k`, outputs:
    - WCSS (Within-Cluster Sum of Squares)
    - My Silhouette Score (Custom implementation)
    - Real Silhouette Score (Using built-in methods)

Example Output:

k = 2, : WCSS: 152.36870647733903, My Silhouette Score: 0.6813042762896313, Real Silhouette Score: 0.6808136202936815 
k = 3, : WCSS: 78.94084142614602, My Silhouette Score: 0.56003817789559, Real Silhouette Score: 0.5525919445499754 
... 
k = 9, : WCSS: 29.07916573864944, My Silhouette Score: 0.37373491654788776, Real Silhouette Score: 0.3082630415946508

### `part_C2.py`

- **Purpose**: Reproduces the table and graph for determining the optimal `k` for the **Wine dataset**.
- **Outputs**:
  - **Graph**: Elbow method plot and Silhouette Score for different values of `k`.
  - **Table**: For each `k`, outputs:
    - WCSS (Within-Cluster Sum of Squares)
    - My Silhouette Score (Custom implementation)
    - Real Silhouette Score (Using built-in methods)

Example Output:

k = 2, : WCSS: 4543749.614531862, My Silhouette Score: 0.6608922078311726, Real Silhouette Score: 0.6568536504294317 
k = 3, : WCSS: 2370689.686782968, My Silhouette Score: 0.5784191927934976, Real Silhouette Score: 0.571138193786884 
... 
k = 9, : WCSS: 276587.86464382964, My Silhouette Score: 0.5577207885135143, Real Silhouette Score: 0.5431736964173127


### `part_C3.py`

- **Purpose**: Computes the Normalized Mutual Information (NMI) score for determining optimal `k` after performing clustering.
- **Outputs**:
  - **Table**: For each `k`, outputs:
    - My NMI Score (Custom implementation)
    - Real NMI Score (Using built-in methods)
  - Separate results for the **Iris dataset** and **Wine dataset**.

Example Output:

iris dataset: k = 2, : my NMI: 0.5977807714440937, real NMI: 0.6565191143081124
wine dataset: k = 2, : my NMI: 0.42591862667461183, real NMI: 0.42591862667461144 
... 
wine dataset: k = 9, : my NMI: 0.3432272541669471, real NMI: 0.3511695211713766

### `part_D.py`

- **Purpose**: Compares the performance of the **KMeans** and **KMeans++** algorithms and visualizes the results in graphs.
- **Outputs**:
  - **Graph**: Comparison of WCSS and Silhouette Scores for both KMeans and KMeans++ methods.
  - For each `k`, outputs the following for both **KMeans++** and **KMeans**:
    - WCSS (Within-Cluster Sum of Squares)
    - My Silhouette Score (Custom implementation)
    - Real Silhouette Score (Using built-in methods)

Example Output:

iris dataset KMeans++: k = 2, : WCSS: 152.36870647733903, My Silhouette Score: 0.6813042762896313, Real Silhouette Score: 0.6808136202936815 
... 
k = 9, : WCSS: 290111.64183758595, My Silhouette Score: 0.5596885818694388, Real Silhouette Score: 0.5052248272657792
KMeans: k = 9, : WCSS: 276587.86464382964, My Silhouette Score: 0.5577207885135143, Real Silhouette Score: 0.5431736964173127


## Required Python Packages

Ensure you have the following Python packages installed:

- `sklearn`
- `pandas`
- `matplotlib`
- `numpy`
- `random`
- `math`

## How to Run

1. Clone the repository to your local machine.
2. Install the required Python packages.
3. Execute the corresponding Python script to reproduce the tables and graphs:
   - `part_C1.py`: Iris dataset clustering results.
   - `part_C2.py`: Wine dataset clustering results.
   - `part_C3.py`: NMI scores after determining optimal `k`.
   - `part_D.py`: Performance comparison of KMeans vs KMeans++.

## Example Output

- **Elbow Method Graphs**
- **Silhouette Score Tables**
- **NMI Score Tables**
- **Performance Comparison Graphs**

## Conclusion

The project demonstrates the comparison of clustering algorithms (KMeans vs KMeans++) and evaluates their performance on well-known datasets using multiple evaluation metrics such as WCSS, Silhouette Score, and NMI score.
















