import numpy as np
import operator
from Lib.SimilarityMeasures import Cosine

from recommenders.abstract_recommender import AbstractRecommender


class ItemKNN(AbstractRecommender):

    def __init__(self, ratings, fold_id):
        """
        :type ratings: train_ratings
        :param ratings: может быть train_ratings || test_ratings
        """
        self.ratings = ratings
        self.sim_matrix = None
        self.fold_id = fold_id

        self.k = np.inf
        self.threshold = 0

    def train(self):
        super().train(self.ratings)
        cosine = Cosine(self.ratings, load_matrices=True, save_matrices=True, fold_id=self.fold_id)
        self.sim_matrix = cosine.build()

    def predict(self, user_id, item_id, rating=0):
        ratings = self.ratings

        if not ratings.check_user_item(user_id, item_id):
            print("Failure: ", ratings.global_avg, user_id, item_id, rating)

        items = ratings.user_rated_on(user_id)

        sim_list = []
        for j in items:
            if self.sim_matrix[item_id, j] is not None and self.sim_matrix[item_id, j] > self.threshold:
                sim_list.append((self.sim_matrix[item_id, j], j))

        if len(sim_list) == 0:
            print("Failure: ", ratings.global_avg, user_id, item_id, rating)


        if self.k is not np.inf:
            sim_list.sort(key=operator.itemgetter(0), reverse=True)
            sim_list = sim_list[:self.k]

        sim_list = map(operator.itemgetter(1), sim_list)

        dividend = 0
        divisor = 0
        for j in sim_list:
            rv = ratings.rating_user_item(user_id, j)

            dividend += rv * self.sim_matrix[item_id, j]
            divisor += self.sim_matrix[item_id, j]

        try:
            prediction = dividend / divisor
            prediction = ratings.scale(prediction)
            return prediction

        except ZeroDivisionError:
            print("Failure: ", ratings.global_avg, user_id, item_id, rating)
