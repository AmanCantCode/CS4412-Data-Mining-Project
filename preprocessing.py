import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import re

# 1. LOAD DATA
ratings = pd.read_csv("data/ratings.csv").drop_duplicates()
movies = pd.read_csv("data/movies.csv").drop_duplicates()
tags = pd.read_csv("data/tags.csv").drop_duplicates()

# 2. TEMPORAL CLEANING
# Convert Unix Timestamps to Datetime
ratings['timestamp_dt'] = pd.to_datetime(ratings['timestamp'], unit='s')
tags['timestamp_dt'] = pd.to_datetime(tags['timestamp'], unit='s')

# --- 3. MOVIE-LEVEL PREPROCESSING
movie_stats = ratings.groupby('movieId').agg(
    avg_rating=('rating', 'mean'),
    rating_count=('rating', 'count')
).reset_index()

movie_stats = movie_stats[movie_stats['rating_count'] >= 10]
movie_stats['popularity_log'] = np.log1p(movie_stats['rating_count'])

tag_counts = tags.groupby('movieId')['tag'].count().reset_index(name='metadata_richness')
movie_df = movies.merge(movie_stats, on='movieId', how='inner')
movie_df = movie_df.merge(tag_counts, on='movieId', how='left').fillna(0)

genre_dummies = movie_df['genres'].str.get_dummies(sep='|')
movie_df = pd.concat([movie_df, genre_dummies], axis=1)

def extract_year(title):
    year = re.search(r'\((\d{4})\)', title)
    return int(year.group(1)) if year else 0
movie_df['release_year'] = movie_df['title'].apply(extract_year)

# Save Movie File
movie_df.to_csv("data/preprocessed_movies.csv", index=False)

# --- 4. USER-LEVEL FEATURE ENGINEERING (New Discovery Features) ---

# A. Binge Behavior: Ratings per Day - calculate the max ratings a user gave in a single 24-hour period
user_daily = ratings.groupby(['userId', ratings['timestamp_dt'].dt.date]).size().reset_index(name='daily_cnt')
binge_feature = user_daily.groupby('userId')['daily_cnt'].max().reset_index(name='max_daily_ratings')

# B. Seasonality: Favorite Month
ratings['month'] = ratings['timestamp_dt'].dt.month
user_seasonality = ratings.groupby('userId')['month'].agg(lambda x: x.value_counts().index[0]).reset_index(name='peak_month')

# C. Tagging Behavior
user_tags = tags.groupby('userId')['tag'].count().reset_index(name='total_tags_given')

# D. Aggregating General Stats
user_stats = ratings.groupby('userId').agg(
    avg_user_rating=('rating', 'mean'),
    total_ratings_given=('rating', 'count'),
    rating_standard_dev=('rating', 'std')
).reset_index().fillna(0)

# E. Merge User Features
user_df = user_stats.merge(binge_feature, on='userId', how='left')
user_df = user_df.merge(user_seasonality, on='userId', how='left')
user_df = user_df.merge(user_tags, on='userId', how='left').fillna(0)

# --- 5. STANDARDIZE USER FEATURES ---
# Standardize for K-Means (Section 4.4)
scaler = StandardScaler()
u_numeric = ['avg_user_rating', 'total_ratings_given', 'max_daily_ratings', 'total_tags_given']
user_df[u_numeric] = scaler.fit_transform(user_df[u_numeric])

# Save User File
user_df.to_csv("data/preprocessed_users.csv", index=False)

print(f"Preprocessing complete.")
print(f"Movies file: {movie_df.shape}")
print(f"Users file: {user_df.shape}")