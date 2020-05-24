from evaluation.abstract_evaluator import AbstractRecommenderEvaluator
import numpy as np


class RMSEEvaluator(AbstractRecommenderEvaluator):
    def __init__(self):
        super().__init__()

    def evaluate(self, predictions, verbose=True):
        if not predictions:
            raise ValueError('Prediction list is empty.')

        mse = np.mean([float((true_r - est) ** 2)
                       for (_, _, true_r, est, _) in predictions])
        rmse_ = np.sqrt(mse)

        if verbose:
            print('RMSE: {0:1.4f}'.format(rmse_))

        return rmse_