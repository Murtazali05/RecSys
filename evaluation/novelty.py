from evaluation.abstract_evaluator import AbstractRecommenderEvaluator
import numpy as np


class NoveltyEvaluator(AbstractRecommenderEvaluator):
    def __init__(self):
        super().__init__()

    def evaluate(self, topNPredicted, rankings):
        n = 0
        total = 0
        for userID in topNPredicted.keys():
            for rating in topNPredicted[userID]:
                movieID = rating[0]
                rank = rankings[movieID]
                total += rank
                n += 1
        return total / n