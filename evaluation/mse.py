from evaluation.abstract_evaluator import AbstractRecommenderEvaluator
import numpy as np


class RMSEEvaluator(AbstractRecommenderEvaluator):
    def __init__(self):
        super().__init__()

    def evaluate(self, predictions, verbose=True):
        if not predictions:
            raise ValueError('Prediction list is empty.')

        mse_ = np.mean([float((true_r - est) ** 2)
                        for (_, _, true_r, est, _) in predictions])

        if verbose:
            print('MSE: {0:1.4f}'.format(mse_))

        return mse_
