from abc import abstractmethod


class AbstractRecommender(object):
    @abstractmethod
    def train(self, train_ratings):
        pass

    @abstractmethod
    def predict(self, user_id, item_id):
        pass