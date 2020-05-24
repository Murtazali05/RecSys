from abc import abstractmethod


class AbstractRecommenderEvaluator:
    def __init__(self):
        pass

    @abstractmethod
    def evaluate(self, predictions):
        pass