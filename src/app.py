import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#load data
movies = pd.read_csv('../data/raw/tmdb_5000_movies.csv')
credits = pd.read_csv('../data/raw/tmdb_5000_credits.csv')

#merge both dataset
df_movies = movies.merge(credits, on='title') 

#select some feature
df_movies = df_movies[['movie_id','title','overview','genres','keywords','cast','crew']] 

#remove 3 empty value of overview
df_movies.dropna(inplace = True)

#transform the json to list
import ast

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

df_movies['genres'] = df_movies['genres'].apply(convert)
df_movies['keywords'] = df_movies['keywords'].apply(convert)

#transform the json to list limit to 3 value
def convert3(obj):
    L = []
    count = 0
    for i in ast.literal_eval(obj):
        if count < 3:
            L.append(i['name'])
        count +=1  
    return L

df_movies['cast'] = df_movies['cast'].apply(convert3)

#get the director of the moview from crew
def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

df_movies['crew'] = df_movies['crew'].apply(fetch_director)

#collapse the value of the feature, remove space
def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
        
    return " ".join(L1)

df_movies['cast'] = df_movies['cast'].apply(collapse)
df_movies['crew'] = df_movies['crew'].apply(collapse)
df_movies['genres'] = df_movies['genres'].apply(collapse)
df_movies['keywords'] = df_movies['keywords'].apply(collapse)

#create new feature join the clean feature
df_movies['tags'] = df_movies['overview']+df_movies['genres']+df_movies['keywords']+df_movies['cast']+df_movies['crew']

#select again only some feature join
new_df = df_movies[['movie_id','title','tags']]

#Text vectorization 
cv = CountVectorizer(max_features=5000 ,stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
 
similarity = cosine_similarity(vectors)
sorted(list(enumerate(similarity[0])),reverse =True , key = lambda x:x[1])[1:6]

#Finally, create a recommendation function based on the cosine_similarity. 
#This function should recommend the 5 most similar movies.
def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0] ##fetching the movie index
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate( distances)),reverse =True , key = lambda x:x[1])[1:6]
    
    for i in movie_list:
        print(new_df.iloc[i[0]].title)

recommend('Avatar')