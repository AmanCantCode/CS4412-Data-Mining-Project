
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os

# Create directory for visuals
os.makedirs("preprocessing_visuals", exist_ok=True)

# 1. LOAD DATA - use movie_df from your previous step BEFORE scaling to show the "Raw" state
movie_df = pd.read_csv("data/preprocessed_movies.csv")
user_df = pd.read_csv("data/preprocessed_users.csv")

# --- FIGURE 1: LOG TRANSFORMATION (Handling Skewed Outliers) --- 
# This shows 'popularity_log' was created
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

sns.histplot(movie_df['rating_count'], bins=50, ax=ax1, color='firebrick')
ax1.set_title('BEFORE: Raw Rating Count (Extreme Skew)')
ax1.set_xlabel('Number of Ratings')

sns.histplot(movie_df['popularity_log'], bins=30, ax=ax2, color='forestgreen', kde=True)
ax2.set_title('AFTER: Log-Transformed Popularity (Normalized)')
ax2.set_xlabel('Log(Rating Count + 1)')

plt.tight_layout()
plt.savefig("preprocessing_visuals/popularity_transformation.png", dpi=300, bbox_inches='tight')

# --- FIGURE 2: THE NEED FOR STANDARDIZATION ---
# This demonstrates the magnitude difference between Tags (1200+) and Ratings (1-5)
raw_features = movie_df[['avg_rating', 'metadata_richness']]
plt.figure(figsize=(10, 6))
sns.boxplot(data=raw_features, palette="Set2")
plt.title('Scale Disparity between Ratings and Tag Counts')
plt.ylabel('Raw Value Magnitude')
plt.savefig("preprocessing_visuals/scale_disparity.png", dpi=300, bbox_inches='tight')

# --- FIGURE 3: POST-STANDARDIZATION BALANCE ---
# Apply scaling to show the balanced state
scaler = StandardScaler()
scaled_movie_data = movie_df[['avg_rating', 'popularity_log', 'metadata_richness']]
scaled_vals = scaler.fit_transform(scaled_movie_data)
scaled_df = pd.DataFrame(scaled_vals, columns=['Rating', 'Popularity', 'Tags'])

plt.figure(figsize=(10, 6))
sns.kdeplot(data=scaled_df)
plt.title('Features Post-Standardization (Mean=0, Std=1)')
plt.xlabel('Z-Score')
plt.savefig("preprocessing_visuals/standardized_features.png", dpi=300, bbox_inches='tight')

# --- FIGURE 4: USER TEMPORAL DISCOVERY (Section 3.2/3.3) ---
# Show the binge behavior distribution
plt.figure(figsize=(10, 6))
sns.histplot(user_df['max_daily_ratings'], bins=50, color='purple', kde=True)
plt.title('Discovery: Identifying Binge-Rating Behavior')
plt.xlabel('Standardized Max Daily Ratings')
plt.savefig("preprocessing_visuals/user_binge_dist.png", dpi=300, bbox_inches='tight')

# --- FIGURE 5: FEATURE CORRELATION 1 ---
plt.figure(figsize=(12, 10))
corr = movie_df[['avg_rating', 'rating_count', 'metadata_richness', 'release_year']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation of Derived Movie Features')
plt.savefig("preprocessing_visuals/feature_heatmap.png", dpi=300, bbox_inches='tight')

print("Preprocessing visuals saved in 'preprocessing_visuals/' folder.")

# --- FIGURE 6: FEATURE CORRELATION 2 ---
corr_features = ["avg_rating", "popularity_log", "metadata_richness", "release_year"]
corr_matrix = movie_df[corr_features].corr()

plt.figure(figsize=(6,5))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.savefig("preprocessing_visuals/feature_correlation_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()

# --- FIGURE 7: Rating Distribution ---
df = pd.read_csv("data/preprocessed_movies_2.csv")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df['avg_rating'], bins=30, kde=True, ax=ax1, color='teal')
ax1.set_title('Distribution of Average Ratings')
sns.boxplot(x=df['avg_rating'], ax=ax2, color='teal')
ax2.set_title('Box Plot of Average Ratings')
plt.savefig("report_images/TESTING.png", dpi=300, bbox_inches='tight')
# (Box Plot is not as critical since ratings are bounded by 0.5 - 5.0)