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

### 1. Installation
First, clone the repository and install the necessary dependencies using the requirements file:

    pip install -r requirements.txt

### 2. Create Data Directory
Since the data files are excluded from version control (via .gitignore), you will need to manually create a `data/` directory:

    mkdir data

### 3. Download and Extract Dataset
Download the MovieLens 32M dataset from:  
https://grouplens.org/datasets/movielens/32m/

Download the file `ml-32m.zip` and extract its contents into the `data/` directory.

The `data/` directory should contain the following CSV files after extraction:
- `ratings.csv`
- `movies.csv`
- `tags.csv`
- `links.csv`

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

### Phase 4: Classification & Validation

#### Movie Decision Tree
Run the decision tree to validate movie clusters and identify feature importance:

    python movies_d_tree.py

#### User Decision Tree
Run the decision tree to validate user archetypes and identify what separates them:

    python users_d_tree.py

### Phase 5: Temporal Analysis & Anomaly Detection

Run the user behavior analysis script to examine session patterns, rating recency, lifespan, and detect anomalies:

    python user_behavior.py

-------------------------------------------------------------------
## Output and Visualizations

All scripts are designed to automatically create the necessary directories and save visualizations without any manual intervention. As you run each script, the following dedicated folders will be created automatically in the project root:

| Script | Generated Folder | Contents |
|--------|------------------|----------|
| `EDA.py` | `original_eda_graphs/` | Distribution plots after preprocessing |
| `preprocess_visual.py` | `preprocessing_visuals/` | Distribution plots after preprocessing |
| `clustering_movies.py` | `movie_graphs/` | PCA scatter plots, elbow curves, rating distributions |
| `clustering_users.py` | `user_graphs/` | User cluster visualizations, heatmaps |
| `movies_d_tree.py` | `movie_graphs/` | Movie decision tree visualization |
| `users_d_tree.py` | `user_graphs/` | User decision tree visualization |
| `user_behavior.py` | `user_lifecycle/` | Session analysis plots, temporal patterns, LOF |

**Documentation:** The `docs/` folder contains all project reports from M1 to M4, including the presentation slides.

No additional setup is required. Each script checks for the existence of its required output directory and creates it automatically if missing. All visualizations are saved at 300 DPI for publication quality.

-------------------------------------------------------------------
## Script Execution Order

For a complete run of the entire pipeline, execute the scripts in the following order:

1. `python EDA.py`
2. `python preprocessing.py`
3. `python preprocess_visual.py`
4. `python clustering_movies.py`
5. `python clustering_users.py`
6. `python movies_d_tree.py`
7. `python users_d_tree.py`
8. `python user_behavior.py`