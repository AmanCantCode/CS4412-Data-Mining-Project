
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA

GRAPH_DIR = "user_lifecycle"
os.makedirs(GRAPH_DIR, exist_ok=True)

# 1. LOAD DATA
ratings = pd.read_csv("data/ratings.csv")
movies = pd.read_csv("data/movies.csv")
user_clusters = pd.read_csv("data/user_clusters_final.csv")

# Add datetime
ratings['datetime'] = pd.to_datetime(ratings['timestamp'], unit='s')
ratings = ratings.sort_values(['userId', 'datetime'])

# Merge with user types
ratings = ratings.merge(user_clusters[['userId', 'cluster']], on='userId')
cluster_names = {0: "Engaged - Drama Enthusiasts (Heavy)", 1: "Generous - Action Seekers (Light)"}
ratings['user_type'] = ratings['cluster'].map(cluster_names)

print("="*60)
print("USER LIFECYCLE ANALYSIS: How Rating Behavior Evolves")
print("="*60)

# 2. IDENTIFY SESSIONS
# Mark new sessions (2-hour gap)
ratings['time_diff'] = ratings.groupby('userId')['datetime'].diff().dt.total_seconds() / 3600
ratings['new_session'] = (ratings['time_diff'] > 2) | (ratings['time_diff'].isna())
ratings['session_id'] = ratings.groupby('userId')['new_session'].cumsum()

# Get session stats
session_stats = ratings.groupby(['userId', 'session_id', 'user_type']).agg(
    session_length=('rating', 'count'),
    session_start=('datetime', 'min'),
    session_end=('datetime', 'max'),
    avg_session_rating=('rating', 'mean'),
    session_number=('session_id', 'first')
).reset_index()

# Add session order within each user
session_stats['session_order'] = session_stats.groupby('userId').cumcount() + 1

# 3. ANALYZE FIRST SESSION VS LATER SESSIONS
print("\nFIRST SESSION VS LATER SESSIONS")
print("-" * 50)

# Separate first session from others
first_sessions = session_stats[session_stats['session_order'] == 1]
later_sessions = session_stats[session_stats['session_order'] > 1]

# Calculate metrics
first_avg_length = first_sessions['session_length'].mean()
later_avg_length = later_sessions['session_length'].mean()

print(f"\nOverall:")
print(f"  - First session avg length: {first_avg_length:.1f} ratings")
print(f"  - Later sessions avg length: {later_avg_length:.1f} ratings")
print(f"  - Ratio (first/later): {first_avg_length/later_avg_length:.1f}x")

# By user type
print(f"\nBy User Type:")
for user_type in session_stats['user_type'].unique():
    type_first = first_sessions[first_sessions['user_type'] == user_type]['session_length'].mean()
    type_later = later_sessions[later_sessions['user_type'] == user_type]['session_length'].mean()
    ratio = type_first / type_later if type_later > 0 else 0
    print(f"\n  {user_type}:")
    print(f"    - First session: {type_first:.1f} ratings")
    print(f"    - Later sessions: {type_later:.1f} ratings")
    print(f"    - Ratio: {ratio:.1f}x")

# Statistical test
t_stat, p_value = stats.ttest_ind(
    first_sessions['session_length'], 
    later_sessions['session_length']
)
print(f"\nStatistical significance: p-value = {p_value:.4f}")
print(f"  - {'Significant difference' if p_value < 0.05 else 'No significant difference'}")

# 4. ANALYZE HOW SESSIONS EVOLVE OVER TIME
print("\n\nSESSION EVOLUTION OVER TIME")
print("-" * 50)

# Group by session order
session_evolution = session_stats.groupby('session_order').agg(
    avg_length=('session_length', 'mean'),
    median_length=('session_length', 'median'),
    count=('session_id', 'count')
).reset_index()

# Limit to first 20 sessions
session_evolution = session_evolution[session_evolution['session_order'] <= 20]

print("\nSession length by session number (first 10):")
for i in range(1, 11):
    if i in session_evolution['session_order'].values:
        length = session_evolution[session_evolution['session_order'] == i]['avg_length'].values[0]
        print(f"  - Session {i}: {length:.1f} avg ratings")

# By user type
print(f"\nBy User Type (first 5 sessions):")
for user_type in session_stats['user_type'].unique():
    type_data = session_stats[session_stats['user_type'] == user_type]
    type_evolution = type_data.groupby('session_order')['session_length'].mean().reset_index()
    type_evolution = type_evolution[type_evolution['session_order'] <= 5]
    
    print(f"\n  {user_type}:")
    for _, row in type_evolution.iterrows():
        print(f"    - Session {int(row['session_order'])}: {row['session_length']:.1f} ratings")

# 5. ANALYZE RATING QUALITY OVER TIME
print("\n\nRATING QUALITY OVER TIME")
print("-" * 50)

# Average rating by session order
rating_evolution = session_stats.groupby('session_order').agg(
    avg_rating=('avg_session_rating', 'mean')
).reset_index()
rating_evolution = rating_evolution[rating_evolution['session_order'] <= 20]

# First session vs later session rating
first_rating = first_sessions['avg_session_rating'].mean()
later_rating = later_sessions['avg_session_rating'].mean()
change = later_rating - first_rating
arrow = "↑" if change > 0 else "↓"

print(f"\nRating change:")
print(f"  - First session avg rating: {first_rating:.2f} stars")
print(f"  - Later sessions avg rating: {later_rating:.2f} stars")
print(f"  - Change: {arrow} {abs(change):.2f} stars")

# By user type
print(f"\nBy User Type:")
for user_type in session_stats['user_type'].unique():
    type_first = first_sessions[first_sessions['user_type'] == user_type]['avg_session_rating'].mean()
    type_later = later_sessions[later_sessions['user_type'] == user_type]['avg_session_rating'].mean()
    type_change = type_later - type_first
    arrow = "up" if type_change > 0 else "down"
    print(f"\n  {user_type}:")
    print(f"    - First: {type_first:.2f} stars,  Later: {type_later:.2f} stars {arrow} {abs(type_change):.2f} stars")

# 6. VISUALIZATIONS
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: First session vs later sessions
ax1 = axes[0, 0]
data_to_plot = [
    first_sessions['session_length'].clip(upper=100),
    later_sessions['session_length'].clip(upper=100)
]
bp = ax1.boxplot(data_to_plot, labels=['First Session', 'Later Sessions'], patch_artist=True)
bp['boxes'][0].set_facecolor('lightcoral')
bp['boxes'][1].set_facecolor('lightblue')
ax1.set_ylabel('Session Length (# ratings)')
ax1.set_title('First Session vs Later Sessions')
ax1.grid(True, alpha=0.3)

# Plot 2: Session length evolution
ax2 = axes[0, 1]
for user_type in session_stats['user_type'].unique():
    type_data = session_stats[session_stats['user_type'] == user_type]
    type_evolution = type_data.groupby('session_order')['session_length'].mean()
    type_evolution = type_evolution[type_evolution.index <= 15]
    ax2.plot(type_evolution.index, type_evolution.values, marker='o', label=user_type)
ax2.set_xlabel('Session Number')
ax2.set_ylabel('Avg Session Length')
ax2.set_title('How Session Length Evolves')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Rating evolution
ax3 = axes[1, 0]
for user_type in session_stats['user_type'].unique():
    type_data = session_stats[session_stats['user_type'] == user_type]
    type_rating = type_data.groupby('session_order')['avg_session_rating'].mean()
    type_rating = type_rating[type_rating.index <= 15]
    ax3.plot(type_rating.index, type_rating.values, marker='o', label=user_type)
ax3.set_xlabel('Session Number')
ax3.set_ylabel('Avg Rating')
ax3.set_title('How Rating Behavior Evolves')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Distribution of first session sizes
ax4 = axes[1, 1]
for user_type in session_stats['user_type'].unique():
    type_first = first_sessions[first_sessions['user_type'] == user_type]['session_length'].clip(upper=200)
    ax4.hist(type_first, alpha=0.5, bins=30, label=user_type, density=True)
ax4.set_xlabel('First Session Length')
ax4.set_ylabel('Density')
ax4.set_title('First Session Size Distribution')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{GRAPH_DIR}/user_lifecycle.png", dpi=300, bbox_inches='tight')
plt.close()

# 7. KEY FINDINGS
print("\n" + "="*60)
print("KEY DISCOVERIES: User Lifecycle Patterns")
print("="*60)

first_vs_later_ratio = first_avg_length / later_avg_length
decline_rate = (first_avg_length - later_avg_length) / first_avg_length * 100

print(f"\nMAJOR FINDINGS:")
print(f"   - First sessions are {first_vs_later_ratio:.1f}x larger than regular sessions")
print(f"   - Users decline by {decline_rate:.0f}% in activity after first session")
print(f"   - Rating {'increases' if change > 0 else 'decreases'} by {abs(change):.2f} stars after first session")

print(f"\nUSER TYPE DIFFERENCES:")
for user_type in session_stats['user_type'].unique():
    type_first = first_sessions[first_sessions['user_type'] == user_type]['session_length'].mean()
    type_later = later_sessions[later_sessions['user_type'] == user_type]['session_length'].mean()
    type_ratio = type_first / type_later
    print(f"\n   {user_type}:")
    print(f"      - First session: {type_first:.0f} ratings")
    print(f"      - Later sessions: {type_later:.0f} ratings")
    print(f"      - Ratio: {type_ratio:.1f}x")

# 8. ADDITIONAL TEMPORAL PATTERNS
print("\n" + "="*60)
print("ADDITIONAL TEMPORAL PATTERNS")
print("="*60)

# Extract release year from movies
movies['release_year'] = movies['title'].str.extract(r'\((\d{4})\)').astype(float)

# Rating recency (do they rate new or old movies?)
print("\nRATING RECENCY ANALYSIS")
print("-" * 40)

ratings_with_year = ratings.merge(movies[['movieId', 'release_year']], on='movieId')
ratings_with_year['years_after_release'] = ratings_with_year['datetime'].dt.year - ratings_with_year['release_year']

recency_by_type = ratings_with_year.groupby('user_type')['years_after_release'].mean()
print("\nHow soon after release do users rate?")
for user_type, years in recency_by_type.items():
    print(f"  - {user_type}: {years:.1f} years after release")

# User lifespan (time between first and last rating)
print("\nUSER LIFESPAN ANALYSIS")
print("-" * 40)

user_lifespan = ratings.groupby('userId')['datetime'].agg(['min', 'max'])
user_lifespan['lifespan_days'] = (user_lifespan['max'] - user_lifespan['min']).dt.days
user_lifespan = user_lifespan.reset_index()

# Merge with user types
user_lifespan = user_lifespan.merge(user_clusters[['userId', 'cluster']], on='userId')
user_lifespan['user_type'] = user_lifespan['cluster'].map(cluster_names)

lifespan_by_type = user_lifespan.groupby('user_type')['lifespan_days'].mean()
print("\nHow long do users stay active?")
for user_type, days in lifespan_by_type.items():
    print(f"  - {user_type}: {days:.0f} days")

# Visualize additional patterns
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Recency plot
ax1 = axes[0]
recency_by_type.plot(kind='bar', ax=ax1, color=['red', 'blue'])
ax1.set_title('How Soon After Release Do Users Rate?')
ax1.set_xlabel('User Type')
ax1.set_ylabel('Years After Release')
ax1.set_xticklabels(recency_by_type.index, rotation=0)
ax1.grid(True, alpha=0.3)

# Lifespan plot
ax2 = axes[1]
lifespan_by_type.plot(kind='bar', ax=ax2, color=['red', 'blue'])
ax2.set_title('User Lifespan on Platform')
ax2.set_xlabel('User Type')
ax2.set_ylabel('Days Active')
ax2.set_xticklabels(lifespan_by_type.index, rotation=0)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{GRAPH_DIR}/temporal_patterns.png", dpi=300, bbox_inches='tight')
plt.close()

# 9. SAVE RESULTS
session_stats.to_csv("data/user_lifecycle_analysis.csv", index=False)
print("\nAnalysis complete. Results saved to data/user_lifecycle_analysis.csv")

# 10. ANOMALY DETECTION WITH LOCAL OUTLIER FACTOR
print("\nANOMALY DETECTION: Local Outlier Factor")
print("-" * 40)

# Select behavioral features for anomaly detection
anomaly_features = ['actual_avg_rating', 'actual_rating_std', 'actual_total_ratings', 
                    'actual_tags_given', 'actual_max_daily']

# Get the actual values from user_clusters (already loaded)
X_anomaly = user_clusters[anomaly_features].fillna(0)

# Run LOF with auto contamination
lof = LocalOutlierFactor(n_neighbors=20, contamination='auto')
user_clusters['lof_outlier'] = lof.fit_predict(X_anomaly)
user_clusters['lof_score'] = lof.negative_outlier_factor_

# Count outliers
n_outliers = (user_clusters['lof_outlier'] == -1).sum()
pct_outliers = n_outliers / len(user_clusters) * 100

print(f"\nOutliers detected: {n_outliers} users ({pct_outliers:.1f}%)")

# Outliers by user type
outlier_by_cluster = user_clusters.groupby('cluster')['lof_outlier'].agg(
    total='count',
    outliers=lambda x: (x == -1).sum()
)
outlier_by_cluster['pct'] = outlier_by_cluster['outliers'] / outlier_by_cluster['total'] * 100
print("\nOutliers by user type:")
print(outlier_by_cluster)

# Characterize outliers
outliers = user_clusters[user_clusters['lof_outlier'] == -1]
normal = user_clusters[user_clusters['lof_outlier'] == 1]

print("\nOutlier Characteristics:")
for feature in anomaly_features:
    outlier_mean = outliers[feature].mean()
    normal_mean = normal[feature].mean()
    print(f"  {feature}: {outlier_mean:.2f} (outliers) vs {normal_mean:.2f} (normal)")

# Save anomaly results
user_clusters.to_csv("data/user_anomaly_results.csv", index=False)
print("\nAnomaly results saved to data/user_anomaly_results.csv")

# Generate LOF Visuals

# PCA to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_anomaly)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[user_clusters['lof_outlier']==1, 0],
            X_pca[user_clusters['lof_outlier']==1, 1],
            s=5, alpha=0.3, label='Normal Users')

plt.scatter(X_pca[user_clusters['lof_outlier']==-1, 0],
            X_pca[user_clusters['lof_outlier']==-1, 1],
            s=8, label='LOF Outliers')

plt.xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('LOF Outliers in User Behavior Space')
plt.legend()
plt.savefig(f'{GRAPH_DIR}/lof_pca_scatter.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(user_clusters['lof_score'], bins=100, color='steelblue', edgecolor='black', alpha=0.7)
threshold = user_clusters[user_clusters['lof_outlier'] == -1]['lof_score'].min()
plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label='Outlier Threshold')
plt.xlabel('LOF Score (lower = more anomalous)')
plt.ylabel('Number of Users')
plt.title('Distribution of LOF Scores')
plt.legend()
plt.savefig(f'{GRAPH_DIR}/lof_score_distribution.png', dpi=300, bbox_inches='tight')
plt.close()


means_out = [outliers[f].mean() for f in anomaly_features]
means_norm = [normal[f].mean() for f in anomaly_features]

x = np.arange(len(anomaly_features))
width = 0.35

plt.figure(figsize=(10,6))
plt.bar(x - width/2, means_norm, width, label='Normal Users')
plt.bar(x + width/2, means_out, width, label='Outliers')

plt.xticks(x, anomaly_features, rotation=45)
plt.title('Behavior Comparison: Normal vs LOF Outliers')
plt.legend()
plt.tight_layout()
plt.savefig(f'{GRAPH_DIR}/lof_feature_comparison.png', dpi=300, bbox_inches='tight')
plt.show()