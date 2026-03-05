# This file helps in performing EDA on the ORIGINAL raw dataset before 
# applying any preprocessing and data mining techniques

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create directory for original visuals
os.makedirs("original_eda_graphs", exist_ok=True)

# 1. LOAD ORIGINAL CSVs
ratings_raw = pd.read_csv("data/ratings.csv")
movies_raw = pd.read_csv("data/movies.csv")
tags_raw = pd.read_csv("data/tags.csv")

# --- RAW UNIVARIATE ANALYSIS ---

# Distribution of Raw Ratings (0.5 to 5.0)
plt.figure(figsize=(10, 6))
sns.countplot(x='rating', data=ratings_raw, color='skyblue')
plt.title('Distribution of Raw User Ratings (0.5-5.0)')
plt.xlabel('Star Rating')
plt.ylabel('Total Count')
plt.savefig("original_eda_graphs/3_1_raw_ratings_count.png")

# Identifying Genres (Exploding the pipe-separated list)
genres_split = movies_raw['genres'].str.get_dummies(sep='|')
genre_counts = genres_split.sum().sort_values(ascending=False)

plt.figure(figsize=(12, 8))
genre_counts.plot(kind='bar', color='salmon')
plt.title('Frequency of Movie Genres in Original Dataset')
plt.ylabel('Number of Movies')
plt.savefig("original_eda_graphs/3_1_raw_genre_freq.png", bbox_inches='tight')

# --- RAW BIVARIATE ANALYSIS ---

# Merge to see relationship between Genre and Rating (Raw)
# (Taking a sample to speed up calculation if needed)
raw_merged = ratings_raw.merge(movies_raw, on='movieId')

# Question: Do certain genres naturally get higher raw ratings?
# We'll check the top 5 genres
top_5_genres = genre_counts.head(5).index.tolist()
genre_rating_data = []

for genre in top_5_genres:
    subset = raw_merged[raw_merged['genres'].str.contains(genre)]
    genre_rating_data.append(pd.DataFrame({'Genre': [genre]*len(subset), 'Rating': subset['rating']}))

plot_df = pd.concat(genre_rating_data)

plt.figure(figsize=(10, 6))
sns.boxplot(x='Genre', y='Rating', data=plot_df, palette='Set3')
plt.title('Raw Rating Distribution by Top Genres')
plt.savefig("original_eda_graphs/3_2_genre_vs_rating_raw.png", dpi = 300)

# --- DATA QUALITY / SUSPICIOUS DATA ---

# Check for "Cold Start" Problem (Movies with very few ratings)
movie_popularity = ratings_raw.groupby('movieId').size()
plt.figure(figsize=(10, 6))
plt.hist(movie_popularity, bins=100, range=(0, 500), color='gray')
plt.title('Data Quality: The Long Tail (Count of Ratings per Movie)')
plt.xlabel('Number of Ratings')
plt.ylabel('Number of Movies')
plt.savefig("original_eda_graphs/3_3_raw_popularity_quality.png")

print("\nOriginal Raw EDA visuals generated in 'original_eda_graphs/'.")
