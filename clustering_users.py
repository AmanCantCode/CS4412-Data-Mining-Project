import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# =========================
# USER CLUSTERING ANALYSIS
# =========================

GRAPH_DIR = "user_graphs"
os.makedirs(GRAPH_DIR, exist_ok=True)

# 1. LOAD PREPROCESSED USERS

user_df = pd.read_csv("data/preprocessed_users.csv")

# Load RAW ratings to get ACTUAL values
ratings_raw = pd.read_csv("data/ratings.csv")
movies_raw = pd.read_csv("data/movies.csv")

# Calculate ACTUAL user stats from raw data (not standardized!)
user_actual_stats = ratings_raw.groupby('userId').agg(
    actual_avg_rating=('rating', 'mean'),
    actual_total_ratings=('rating', 'count'),
    actual_rating_std=('rating', 'std'),
    actual_max_daily=('rating', lambda x: x.groupby(pd.to_datetime(ratings_raw.loc[x.index, 'timestamp'], unit='s').dt.date).count().max())
).reset_index().fillna(0)

# Get actual tag counts
tags_raw = pd.read_csv("data/tags.csv")
user_actual_tags = tags_raw.groupby('userId')['tag'].count().reset_index(name='actual_tags_given')

# Merge actual values
user_actual = user_actual_stats.merge(user_actual_tags, on='userId', how='left').fillna(0)


# 2. LOAD DATA FOR GENRE FEATURES
ratings = pd.read_csv("data/ratings.csv")
movies = pd.read_csv("data/movies.csv")


# 3. BUILD GENRE PREFERENCES
ratings_movies = ratings.merge(movies, on='movieId')

# One-hot encode genres
genre_dummies = ratings_movies['genres'].str.get_dummies(sep='|')

# Aggregate per user (frequency-based, interpretable)
user_genre_prefs = genre_dummies.groupby(ratings_movies['userId']).mean().reset_index()


# 4. MERGE EVERYTHING
# For clustering (using preprocessed/scaled features)
user_df = user_df.merge(user_genre_prefs, on='userId', how='left').fillna(0)

# For interpretation (using ACTUAL values)
user_interpret = user_actual.merge(user_genre_prefs, on='userId', how='left').fillna(0)


# 5. PREPARE FEATURES FOR CLUSTERING
X = user_df.drop(columns=['userId', 'peak_month'], errors='ignore')

# Scale ONLY for clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 6. FIND OPTIMAL K
print("\n--- Silhouette Scores ---")
sil_scores = {}

for k in range(2, 6):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    sil_scores[k] = score
    print(f"k={k}, silhouette={score:.3f}")

optimal_k = 2
print(f"\nUsing k = {optimal_k}")


# 7. FINAL MODEL
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
user_df['cluster'] = kmeans.fit_predict(X_scaled)

# Assign clusters to interpretation dataframe
user_interpret['cluster'] = user_df['cluster']


# 8. PCA VISUALIZATION
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

print("User PCA Explained Variance Ratio:")
print(pca.explained_variance_ratio_)
print(f"Cumulative variance for first 2 components: {pca.explained_variance_ratio_.sum():.3f}")

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=user_df['cluster'], cmap='viridis', alpha=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title(f'User Clusters (k={optimal_k})')
plt.colorbar(scatter)
plt.savefig(f"{GRAPH_DIR}/user_clusters_q2.png", dpi=300)
plt.close()

# 9. INTERPRET CLUSTERS
print("\n" + "="*60)
print("USER CLUSTER PROFILES")
print("="*60)

# Define genre columns
genre_cols = [c for c in user_genre_prefs.columns if c != 'userId']

for cluster in range(optimal_k):
    cluster_data = user_interpret[user_interpret['cluster'] == cluster]
    
    print(f"\nCLUSTER {cluster} ({len(cluster_data)} users, {len(cluster_data)/len(user_interpret)*100:.1f}%):")
    
    # Rating behavior (ACTUAL values)
    print(f"\nRATING BEHAVIOR:")
    print(f"  - Average rating given: {cluster_data['actual_avg_rating'].mean():.2f} ★")
    print(f"  - Rating consistency: ±{cluster_data['actual_rating_std'].mean():.2f} stars")
    print(f"  - Total ratings per user: {cluster_data['actual_total_ratings'].mean():.0f}")
    
    # Activity patterns
    print(f"\nACTIVITY PATTERNS:")
    print(f"  - Max ratings in a day: {cluster_data['actual_max_daily'].mean():.1f}")
    print(f"  - Tags per user: {cluster_data['actual_tags_given'].mean():.1f}")
    print(f"  - Users who tag: {(cluster_data['actual_tags_given'] > 0).mean()*100:.1f}%")
    
    # Genre preferences (as percentages)
    print(f"\nTOP GENRES (% of viewing):")
    genre_means = cluster_data[genre_cols].mean().sort_values(ascending=False)
    for genre, pct in genre_means.head(5).items():
        if pct > 0.01:  # Only show if >1%
            print(f"  - {genre}: {pct*100:.1f}%")


# 10. CREATE SUMMARY TABLE
print("\n" + "="*60)
print("CLUSTER COMPARISON SUMMARY")
print("="*60)

summary = []
for cluster in range(optimal_k):
    cluster_data = user_interpret[user_interpret['cluster'] == cluster]
    
    summary.append({
        'Cluster': cluster,
        'Size': f"{len(cluster_data):,}",
        'Avg Rating': f"{cluster_data['actual_avg_rating'].mean():.2f}★",
        'Total Ratings': f"{cluster_data['actual_total_ratings'].mean():.0f}",
        'Binge Score': f"{cluster_data['actual_max_daily'].mean():.1f}/day",
        'Tags/User': f"{cluster_data['actual_tags_given'].mean():.1f}",
        'Top Genre': f"{cluster_data[genre_cols].mean().idxmax()}"
    })

summary_df = pd.DataFrame(summary)
print("\n", summary_df.to_string(index=False))


# 11. SAVE RESULTS
user_interpret.to_csv("data/user_clusters_final.csv", index=False)
print("\nUser clustering complete. Results saved to user_clusters_final.csv")

# =========================
# USER CORRELATION ANALYSIS
# =========================

existing_factors = ['total_ratings_given', 'avg_user_rating', 'rating_standard_dev', 
           'total_tags_given', 'max_daily_ratings']

# Find correlation on existing columns
if len(existing_factors) > 1:
    corr_matrix = user_df[existing_factors].corr()
    print("\nRELATIONSHIP BETWEEN USER FACTORS")
    print("-" * 40)
    print(corr_matrix.round(2))
else:
    print("Not enough factors found for correlation analysis")

# =========================
# BEHAVIORAL METRICS BY CLUSTER
# =========================

# 1. Are power users more diverse in their genre tastes?
print("\nAre power users more diverse in their genre tastes?")
user_df['genre_diversity'] = (user_df[genre_cols] > 0).sum(axis=1)
print("Genre diversity:")
print(user_df.groupby('cluster')['genre_diversity'].mean())

# 2. Do casual users have more extreme ratings (all 5s and 1s)?
print("\nDo casual users have more extreme ratings (all 5s and 1s)?")
user_df['extremity_score'] = user_df['avg_user_rating'].apply(lambda x: abs(x - 3))
print("Rating extremity:")
print(user_df.groupby('cluster')['extremity_score'].mean())


# 3. Do power users rate older/obscure movies?
movies_raw['release_year'] = movies_raw['title'].str.extract(r'\((\d{4})\)').astype(float)

print("\nDo power users rate older/obscure movies?")
user_avg_release = ratings.merge(movies_raw[['movieId', 'release_year']], on='movieId')
user_avg_release = user_avg_release.groupby('userId')['release_year'].mean().reset_index()
user_df = user_df.merge(user_avg_release, on='userId')
print("Avg movie age rated:")
print(user_df.groupby('cluster')['release_year'].mean())

# =========================
# USER CROSS DOMAIN ANALYSIS
# =========================

# Load the movie clusters
movie_clusters = pd.read_csv("data/k4_movie_clusters.csv")
print("Movie cluster distribution:")
print(movie_clusters['cluster'].value_counts().sort_index())

# Get user ratings with movie cluster info
ratings_with_movie_cluster = ratings.merge(
    movie_clusters[['movieId', 'cluster']], 
    on='movieId'
)

# Rename to avoid confusion
ratings_with_movie_cluster = ratings_with_movie_cluster.rename(
    columns={'cluster': 'movie_cluster'}
)

# Merge with user types
ratings_with_both = ratings_with_movie_cluster.merge(
    user_interpret[['userId', 'cluster']], 
    on='userId'
)

# Rename user cluster column
ratings_with_both = ratings_with_both.rename(
    columns={'cluster': 'user_cluster'}
)

# Verify both columns exist and have correct values
print("\nUser cluster values:", ratings_with_both['user_cluster'].unique())
print("Movie cluster values:", ratings_with_both['movie_cluster'].unique())

# Create matrix: User Type vs Movie Cluster (average rating)
user_movie_matrix = ratings_with_both.groupby(
    ['user_cluster', 'movie_cluster']
)['rating'].mean().unstack()

print("\n" + "="*60)
print("USER TYPE x MOVIE CLUSTER MATRIX")
print("="*60)
print("Rows: User Types (0=Engaged Raters, 1=Generous Raters)")
print("Columns: Movie Clusters (0=Long Tail, 1=Popular Hits, 2=Mid-Tier, 3=Classics)")
print(user_movie_matrix.round(2))

# Visualize as heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(user_movie_matrix, annot=True, cmap='RdYlBu', center=3.5, fmt='.2f')
plt.xlabel('Movie Cluster')
plt.ylabel('User Type')
plt.title('User Types x Movie Clusters: Average Rating')
plt.savefig(f"{GRAPH_DIR}/user_movie_matrix.png", dpi=300, bbox_inches='tight')
plt.close()