from evaluation.abstract_evaluator import AbstractRecommenderEvaluator
import numpy as np


class DiversityEvaluator(AbstractRecommenderEvaluator):
    def __init__(self):
        super().__init__()

    def diversity(self, topNPredicted, simsAlgo):
        n = 0
        total = 0
        simsMatrix = simsAlgo.compute_similarities()
        for userID in topNPredicted.keys():
            pairs = np.itertools.combinations(topNPredicted[userID], 2)
            for pair in pairs:
                book1 = pair[0][0]
                book2 = pair[1][0]
                innerID1 = simsAlgo.trainset.to_inner_iid(str(book1))
                innerID2 = simsAlgo.trainset.to_inner_iid(str(book2))
                similarity = simsMatrix[innerID1][innerID2]
                total += similarity
                n += 1

        S = total / n
        return (1 - S)
