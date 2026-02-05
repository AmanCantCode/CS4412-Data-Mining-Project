import pandas as pd
import numpy as np

movies = pd.read_csv('data/movies.csv')
ratings = pd.read_csv('data/ratings.csv')
tags = pd.read_csv('data/tags.csv')
links = pd.read_csv('data/links.csv')
ratings_by_m_id = ratings.sort_values(by=['movieId'], ascending=True)


print(len(movies))
print(len(ratings))
print(len(tags))
print(len(links))

# print(movies.head)
# print(ratings_by_m_id.head)