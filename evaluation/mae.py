from evaluation.abstract_evaluator import AbstractRecommenderEvaluator
import numpy as np


class MAEEvaluator(AbstractRecommenderEvaluator):
    def __init__(self):
        super().__init__()

    def evaluate(self, predictions, verbose=True):
        if not predictions:
            raise ValueError('Prediction list is empty.')

        mae_ = np.mean([float(abs(true_r - est))
                        for (_, _, true_r, est, _) in predictions])

        if verbose:
            print('MAE:  {0:1.4f}'.format(mae_))

        return mae_
