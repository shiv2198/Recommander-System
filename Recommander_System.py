import pandas as pd
import numpy as np

r_cols = ['user_id','movie_id','rating']
ratings = pd.read_csv('D:/SHIVANSH/Machine Learning/Machine_Learning_AZ/Recommander System/ml-100k/u.data', sep='\t', names=r_cols, usecols=range(3) )

m_cols = ['movie_id', 'title']
movies = pd.read_csv('D:/SHIVANSH/Machine Learning/Machine_Learning_AZ/Recommander System/ml-100k/u.item', sep='|', names=m_cols, usecols=range(2), encoding = 'latin1')

ratings = pd.merge(movies, ratings)

movie_ratings = ratings.pivot_table(index=['user_id'], columns=['title'], values='rating')
df = pd.DataFrame(movie_ratings)

starWarsRatings = movie_ratings['Star Wars (1977)']

similarMovies = movie_ratings.corrwith(starWarsRatings)
similarMovies = similarMovies.dropna()
similarMovies.head(10)

sm = similarMovies.order(ascending = False)

movieStats = ratings.groupby('title').agg({'rating' : [np.size , np.mean]})

popular_movies = movieStats['rating']['size']>=100

movieStats[popular_movies].sort([('rating','mean')] , ascending = False)

df = movieStats[popular_movies].join(pd.DataFrame(similarMovies, columns=['similarity']))

df = df.sort(['similarity'], ascending=False)
