# CS4412 Data Mining Project
Project by Aman Bhayani

## Overview
This project analyzes the MovieLens dataset to discover patterns and insights using data mining techniques. The analysis includes clustering of movies and users, classification for validation, temporal analysis of user behavior, and anomaly detection.

## Data Source Link
https://grouplens.org/datasets/movielens/

## Technologies Used
- Python 3.x
- Data Manipulation: Pandas, NumPy
- Machine Learning: Scikit-Learn (K-Means, PCA, StandardScaler, DecisionTreeClassifier, LocalOutlierFactor)
- Visualization: Matplotlib, Seaborn

-------------------------------------------------------------------
## Getting Started

1. Installation:
First, clone the repository and install the necessary dependencies using the requirements file:

    pip install -r requirements.txt

-------------------------------------------------------------------
## Data Mining Pipeline

### Phase 1: Exploratory Data Analysis (EDA)
Before any processing occurs, run the EDA script to understand the raw distributions, identify missing values, and view initial correlations:

    python EDA.py

### Phase 2: Preprocessing & Feature Engineering
Run the preprocessing script to clean the data, handle outliers (Log-transformation), and engineer new features (Binge behavior, Metadata richness). After that, run the visualizer to see how the data distributions have changed:

    python preprocessing.py
    python preprocess_visual.py

### Phase 3: Clustering (M2)

#### Movie Clustering
Execute the clustering script to apply PCA and K-Means. This will identify the "Archetypes" of movies (e.g., All-Time Classics vs. The Long Tail) and generate Silhouette validation plots:

    python clustering_movies.py

#### User Clustering
Execute the user clustering script to identify user archetypes based on behavioral features:

    python clustering_users.py

### Phase 4: Classification & Validation (M3 Expansion)

#### Movie Decision Tree
Run the decision tree to validate movie clusters and identify feature importance:

    python movies_d_tree.py

#### User Decision Tree
Run the decision tree to validate user archetypes and identify what separates them:

    python users_d_tree.py

### Phase 5: Temporal Analysis & Anomaly Detection (M3 Expansion)

Run the user behavior analysis script to examine session patterns, rating recency, lifespan, and detect anomalies:

    python user_behavior.py