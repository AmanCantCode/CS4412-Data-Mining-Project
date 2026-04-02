# D TREE : USERS

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

GRAPH_DIR = "user_graphs"

# Load clustered users
df = pd.read_csv("data/user_clusters_final.csv")

# Drop non-feature columns
# Identify genre columns (same logic as before)
features = [
    'actual_avg_rating',
    'actual_rating_std',
    'actual_total_ratings',
    'actual_max_daily',
    'actual_tags_given'
]

X = df[features]
y = df['cluster']
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
plt.title("Decision Tree Explaining User Archetypes")
plt.savefig(f"{GRAPH_DIR}/user_decision_tree.png", dpi=300, bbox_inches='tight')
plt.close()

# Feature importance
importances = pd.Series(tree.feature_importances_, index=X.columns)

print("\n=== USER ARCHETYPE FEATURE IMPORTANCE ===")
print(importances.sort_values(ascending=False).head(10))