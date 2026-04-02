# D TREE : MOVIES

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
import os

GRAPH_DIR = "movie_graphs"
os.makedirs(GRAPH_DIR, exist_ok=True)

# Load clustered movie data
df = pd.read_csv("data/k4_movie_clusters.csv")

# Identify features
non_genre_cols = ['movieId', 'title', 'genres', 'cluster']
exclude_cols = non_genre_cols + ['avg_rating', 'popularity_log', 'metadata_richness']
genre_cols = [c for c in df.columns if c not in exclude_cols]

features = ['avg_rating', 'popularity_log'] + genre_cols

X = df[features]

# Remove duplicate columns if any
X = X.loc[:, ~X.columns.duplicated()]

y = df['cluster']

# Train Decision Tree
tree = DecisionTreeClassifier(max_depth=4, random_state=42)
tree.fit(X, y)

# Plot tree
plt.figure(figsize=(20,10))
plot_tree(tree,
          feature_names=X.columns,
          class_names=[str(i) for i in sorted(y.unique())],
          filled=True)

plt.title("Decision Tree Explaining Movie Clusters")
plt.savefig(f"{GRAPH_DIR}/movie_decision_tree.png", dpi=300, bbox_inches='tight')
plt.close()

# Feature importance
importances = pd.Series(tree.feature_importances_, index=X.columns)

print("\n=== MOVIE CLUSTER FEATURE IMPORTANCE ===")
print(importances.sort_values(ascending=False).head(10))

y_pred = tree.predict(X)
accuracy = accuracy_score(y, y_pred)
print(f"\n=== MODEL ACCURACY ===")
print(f"Decision Tree Accuracy: {accuracy:.3f}")
print(f"Without metadata_richness: {accuracy:.3f}")