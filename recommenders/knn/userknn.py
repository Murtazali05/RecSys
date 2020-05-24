import numpy as np
import operator
from Lib.SimilarityMeasures import Pearson
from recommenders.abstract_recommender import AbstractRecommender


class UserKNN(AbstractRecommender):

    def __init__(self, ratings, fold_id):
        self.ratings = ratings
        self.sim_matrix = None
        self.fold_id = fold_id

        self.threshold = 0
        self.k = np.inf

    def train(self):
        super().train(self.ratings)
        pearson = Pearson(self.ratings, load_matrices=True, save_matrices=True, fold_id=self.fold_id)
        self.sim_matrix = pearson.build()

    def predict(self, user_id, item_id, rating=0):
        ratings = self.ratings

        if not ratings.check_user_item(user_id, item_id):
            print("Failure: ", ratings.global_avg, user_id, item_id, rating)

        try:
            indexes = ratings.by_item[item_id]
        except KeyError:
            print("Failure: ", ratings.global_avg, user_id, item_id, rating)

        lst = []
        for i in indexes:
            v = ratings.users[i]
            if user_id == v:
                continue

            sim = self.sim_matrix[user_id, v]
            if sim is not None and sim > self.threshold:
                lst.append((self.sim_matrix[user_id, v], v))

        if len(lst) == 0:
            print("Failure: ", ratings.global_avg, user_id, item_id, rating)

        if self.k is not np.inf:
            lst.sort(key=operator.itemgetter(0), reverse=True)
            lst = lst[:self.k]

        lst = map(operator.itemgetter(1), lst)

        dividend = 0
        divisor = 0
        for v in lst:
            rv = ratings.rating_user_item(v, item_id)

            dividend += (rv - ratings.user_ratings_avg[v]) * self.sim_matrix[user_id, v]
            divisor += self.sim_matrix[user_id, v]

        try:
            prediction = ratings.user_ratings_avg[user_id] + (dividend / divisor)
            prediction = ratings.scale(prediction)
            return prediction
        except ZeroDivisionError:
            print("Failure: ", ratings.global_avg, user_id, item_id, rating)
