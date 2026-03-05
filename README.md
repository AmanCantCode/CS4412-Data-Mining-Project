# CS4412 Data Mining Project
Project by Aman Bhayani

## Overview
This project analyzes the MovieLens dataset to discover patterns and insights using data mining techniques.

## Data Source Link
https://grouplens.org/datasets/movielens/

## Technologies Used
- Python 3.x
- Data Manipulation: Pandas, NumPy
- Machine Learning: Scikit-Learn (K-Means, PCA, StandardScaler)
- Visualization: Matplotlib, Seaborn

##Getting Started
1. Installation:
First, clone the repository and install the necessary dependencies using the requirements file:

pip install -r requirements.txt

2. Exploratory Data Analysis (EDA)
Before any processing occurs, run the EDA script to understand the raw distributions, identify missing values, and view initial correlations:

python EDA.py

3. Preprocessing & Feature Engineering
Run the preprocessing script to clean the data, handle outliers (Log-transformation), and engineer new features (Binge behavior, Metadata richness). After that, run the visualizer to see how the data distributions have changed:

python preprocessing.py
python preprocess_visual.py

4. Data Mining: Movie Clustering
Finally, execute the clustering script to apply PCA and K-Means. This will identify the "Archetypes" of movies (e.g., All-Time Classics vs. The Long Tail) and generate Silhouette validation plots:

python clustering_movies.py