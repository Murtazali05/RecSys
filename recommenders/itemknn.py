import operator

import numpy as np
import pandas as pd
from scipy import spatial


class ItemKNN:
    def __init__(self):
        pass

    def similarity(self):
        # centered cosine
        pass

    def nearest_neighbors(self):
        # KNN
        pass


ratings = pd.read_csv('../datasets/data/ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating'], usecols=range(3))
print(ratings.head())

movieProperties = ratings.groupby('movie_id').agg({'rating': [np.size, np.mean]})
print(movieProperties.head())

movieNumRatings = pd.DataFrame(movieProperties['rating']['size'])
movieNormalizedNumRatings = movieNumRatings.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
print(movieNormalizedNumRatings.head())

movieDict = {}
with open(r'../datasets/data/ml-100k/u.item', encoding="ISO-8859-1") as f:
    temp = ''
    for line in f:
        # line.encode().decode("ISO-8859-1")
        fields = line.rstrip('\n').split('|')
        print(fields)
        movieID = int(fields[0])
        name = fields[1]
        genres = fields[5:25]
        genres = map(int, genres)
        movieDict[movieID] = (name, np.array(list(genres)), movieNormalizedNumRatings.loc[movieID].get('size'),
                              movieProperties.loc[movieID].rating.get('mean'))

print(movieDict[1])


def ComputeDistance(a, b):
    genresA = a[1]
    genresB = b[1]
    genreDistance = spatial.distance.cosine(genresA, genresB)
    popularityA = a[2]
    popularityB = b[2]
    popularityDistance = abs(popularityA - popularityB)
    return genreDistance + popularityDistance


ComputeDistance(movieDict[1], movieDict[4])


def getNeighbors(movieID, K):
    distances = []
    for movie in movieDict:
        if (movie != movieID):
            dist = ComputeDistance(movieDict[movieID], movieDict[movie])
            distances.append((movie, dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(K):
        neighbors.append(distances[x][0])
    return neighbors


K = 10
avgRating = 0

print(movieDict[1], '\n')
neighbors = getNeighbors(1, K)  # Toy Story (1995)
for neighbor in neighbors:
    avgRating += movieDict[neighbor][3]
    print(movieDict[neighbor][0] + " " + str(movieDict[neighbor][3]))

avgRating /= K
