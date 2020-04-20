import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# importing the dataset
movies_dataset = pd.read_csv('../../datasets/data/ml-latest-small/movies.csv')
ratings_dataset = pd.read_csv('../../datasets/data/ml-latest-small/ratings.csv')

# Converting the format of Genre column to a list and then appending to the new list
Genre = []
Genres = {}
for num in range(0, len(movies_dataset)):
    key = movies_dataset.iloc[num]['title']
    value = movies_dataset.iloc[num]['genres'].split('|')
    Genres[key] = value
    Genre.append(value)

# print(Genre)
# print(Genres)

# Making a new column in our original Dataset
movies_dataset['new'] = Genre
# print(movies_dataset.head())

# Getting the year from the movie column
p = re.compile(r"(?:\((\d{4})\))?\s*$")
years = []
for movies in movies_dataset['title']:
    m = p.search(movies)
    year = m.group(1)
    years.append(year)
movies_dataset['year'] = years
# print(movies_dataset.head())

# Deleting the year from the movies title column
movies_name = []
raw = []
for movies in movies_dataset['title']:
    m = p.search(movies)
    year = m.group(0)
    if year:
        new = re.split(year, movies)
    # print(new)
    raw.append(new)
for i in range(len(raw)):
    movies_name.append(raw[i][0][:-2])

# print(movies_dataset.head())

# Making a new column in the dataset having the movie name only in it
movies_dataset['movie_name'] = movies_name

# print(movies_dataset.head())

# Converting the datatype of new column from list to string as required by the function
movies_dataset['new'] = movies_dataset['new'].apply(' '.join)

# print(movies_dataset['new'].head())

# Applying Feature extraction
tfid = TfidfVectorizer(stop_words='english')
# matrix after applying the tfidf
matrix = tfid.fit_transform(movies_dataset['new'])
# print(tfid.idf_)
print(matrix)

# Compute the cosine similarity of every genre
cosine_sim = cosine_similarity(matrix, matrix)
print("cosine_sim:")
print(cosine_sim)
# Making a new series which have two columns in it
# Movie name and movie id
# print(movies_dataset.head(10))
movies_dataset = movies_dataset.reset_index()
# print(movies_dataset.head(10))

titles = movies_dataset['movie_name']
indices = pd.Series(movies_dataset.index, index=movies_dataset['movie_name'])
print('indices:')
print(indices)


# Function to make recommendation to the user
def recommendation(movie):
    result = []
    # Getting the id of the movie for which the user want recommendation
    ind = indices[movie]
    print('We are in method!')
    print(ind)
    # Getting all the similar cosine score for that movie
    sim_scores = list(enumerate(cosine_sim[ind]))
    print(sim_scores)
    # Sorting the list obtained
    # sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Getting all the id of the movies that are related to the movie Entered by the user
    movie_id = [i[0] for i in sim_scores]
    print(movie_id)
    print('The Movie You Should Watched Next Are --')
    print('ID ,   Name ,  Average Ratings , Year ')
    # Varible to print only top 10 movies
    count = 0
    for id in range(0, len(movie_id)):
        # to ensure that the movie entered by the user is doesnot come in his/her recommendation
        print("id=", movie_id[id])
        print(ind.items)
        if ind.index != movie_id[id]:
            ratings = ratings_dataset[ratings_dataset['movieId'] == movie_id[id]]['rating']
            avg_ratings = round(np.mean(ratings), 2)
            # To print only thoese movies which have an average ratings that is more than 3.5
            if avg_ratings > 3.5:
                count += 1
                print('{movie_id[id]} , {titles[movie_id[id]]} ,{avg_ratings}')
                result.append([titles[movie_id[id]], str(avg_ratings)])
            if count >= 10:
                break

    print('Wait!! i am telling your recommendation')
    return result


result = recommendation("Yeh Jawaani Hai Deewani")
print(result)
