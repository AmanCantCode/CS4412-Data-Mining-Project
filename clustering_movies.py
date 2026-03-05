
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from scipy import stats
import os

GRAPH_DIR = "graphs"
os.makedirs(GRAPH_DIR, exist_ok=True)

# 1. Load the NEW Preprocessed Data
df = pd.read_csv("data/preprocessed_movies_2.csv")

# Get actual statistics from original ratings
original_ratings = pd.read_csv("data/ratings.csv")
actual_mean = original_ratings['rating'].mean()
actual_std = original_ratings['rating'].std()

print(f"Statistics from dataset:")
print(f"Mean rating: {actual_mean:.3f}")
print(f"Std deviation: {actual_std:.3f}")

print(f"avg_rating min: {df['avg_rating'].min():.2f}, max: {df['avg_rating'].max():.2f}")
print(f"avg_rating mean: {df['avg_rating'].mean():.2f}, std: {df['avg_rating'].std():.2f}")
print()

# 2. Select Features for Discovery - identify genre columns dynamically
non_genre_cols = [
    'movieId', 'title', 'genres', 'avg_rating', 'rating_count', 
    'popularity_log', 'metadata_richness', 'release_year'
]
genre_cols = [c for c in df.columns if c not in non_genre_cols]

# Numeric features were already scaled in preprocessing
features = ['avg_rating', 'popularity_log', 'metadata_richness'] + genre_cols
X = df[features]

# 3. PCA (Principal Component Analysis)

pca = PCA(n_components=10)
X_pca = pca.fit_transform(X) 

# Apply Silhouette Score to find optimal k
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    print(f"k={k}, silhouette={score:.3f}")

# Apply Elbow Method to find optimal k vakue
wcss = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_pca)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure()
plt.plot(k_range, wcss, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS (Inertia)')
plt.title('Elbow Method For Optimal k')
plt.savefig(f"{GRAPH_DIR}/elbow_plot.png", dpi=300, bbox_inches='tight')
plt.close()

# 4. K-Means Clustering with k=4
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_pca)

# 5. Display the Results
print("\n--- Discovery: Cluster Archetypes ---")

cluster_summary = df.groupby('cluster').agg({
    'avg_rating': 'mean',
    'rating_count': 'mean',
    'metadata_richness': 'mean',
    'release_year': 'mean'
})

for c in range(4):
    # Find top 3 genres that define this cluster
    top_genres = df[df['cluster'] == c][genre_cols].mean().sort_values(ascending=False).head(3)
    
    print(f"\nCluster {c} Profile:")
    print(f"Primary Genres: {', '.join(top_genres.index)}")
    print(f"Avg Rating: {cluster_summary.loc[c, 'avg_rating']:.2f}")
    print(f"Avg Votes: {cluster_summary.loc[c, 'rating_count']:.0f}")
    print(f"Avg Release Year: {int(cluster_summary.loc[c, 'release_year'])}")

# 6. Save final results
df.to_csv("data/k4_movie_clusters.csv", index=False)

# Visualizing the data
# Reduce to 2 components for visualization
pca_vis = PCA(n_components=2)
X_2d = pca_vis.fit_transform(X_pca)

df['PC1'] = X_2d[:, 0]
df['PC2'] = X_2d[:, 1]

plt.figure()
plt.scatter(df['PC1'], df['PC2'], c=df['cluster'])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-Means Clusters (PCA Projection)')
plt.savefig(f"{GRAPH_DIR}/kmeans_pca_clusters.png", dpi=300, bbox_inches='tight')
plt.close()

plt.figure()
plt.scatter(df['rating_count'], df['avg_rating'], c=df['cluster'])
plt.xlabel('Rating Count')
plt.ylabel('Avg Rating')
plt.title('Clusters by Popularity vs Rating')
plt.savefig(f"{GRAPH_DIR}/rating_vs_popularity.png", dpi=300, bbox_inches='tight')
plt.close()

# CLUSTER INTERPRETATION AND SUMMARY TABLE

# Reload the data with cluster assignments
df = pd.read_csv("data/k4_movie_clusters.csv")

# Define cluster names based on patterns (adjust these based on your actual results)
cluster_names = {
    0: "The Long Tail (Average Films)",
    1: "Popular Hits",
    2: "Mid-Tier Successes",
    3: "All-Time Classics"
}

# Comprehensive cluster summary
cluster_summary = []
for c in range(4):
    cluster_movies = df[df['cluster'] == c]
    
    # Get top genres
    top_genres = cluster_movies[genre_cols].mean().sort_values(ascending=False).head(3)
    top_genre_names = [idx for idx in top_genres.index if top_genres.iloc[0] > 0]
    
    # Get example movies
    example_movies = cluster_movies.nlargest(3, 'rating_count')['title'].tolist()
    example_movies = [m.split('(')[0].strip() for m in example_movies]
    
    # Calculate percentile rank
    percentile = stats.percentileofscore(df['avg_rating'], cluster_movies['avg_rating'].mean())
    
    summary = {
        'Cluster': c,
        'Name': cluster_names[c],
        'Size': len(cluster_movies),
        'Primary Genres': ', '.join(top_genre_names[:3]),
        'Avg Rating': f"{cluster_movies['avg_rating'].mean():.2f}",
        'Rating vs All': f"{percentile:.1f}th percentile",
        'Avg Votes': f"{cluster_movies['rating_count'].mean():.0f}",
        'Avg Year': f"{int(cluster_movies['release_year'].mean())}",
        'Examples': ', '.join(example_movies[:2])
    }
    cluster_summary.append(summary)

# Display as table
summary_df = pd.DataFrame(cluster_summary)
print("\n=== NATURAL MOVIE GROUPS DISCOVERED ===\n")
print(summary_df.to_string(index=False))

# Rating distribution by cluster using actual values
plt.figure(figsize=(12, 6))
for cluster in range(4):
    cluster_data = df[df['cluster'] == cluster]['avg_rating']
    plt.hist(cluster_data, alpha=0.5, label=f'Cluster {cluster}: {cluster_names[cluster]}', bins=20, density=True)
plt.xlabel('Average Rating')
plt.ylabel('Number of Movies')
plt.title('Rating Distribution by Cluster')
plt.legend()
plt.savefig(f"{GRAPH_DIR}/rating_distribution_by_cluster.png", dpi=300)
plt.close()