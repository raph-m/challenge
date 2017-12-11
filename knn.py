import numpy as np


class knnClassifier:
    """
    A simple class to implement a knn Classifier
    """

    def __init__(self, labels, a, b, k=10):
        """
        :param labels: (numpy array)
        :param a: (numpy array) sensor_a data
        :param b: (numpy array) sensor_b data
        :param k: (int) number of neighbours to consider for the classifier
        """
        self.a = np.transpose(a)[0]
        self.b = np.transpose(b)[0]
        self.k = k
        self.labels = labels
        assert 0 < k < len(a)
        assert len(a) == len(b)
        assert len(labels) == len(b)
        # self.a_min = -400
        # self.a_max = 300
        # self.b_min = -150
        # self.b_max = 150
        # self.precision = precision
        # self.grid = np.zeros((int((self.a_max - self.a_min)/precision), int((self.b_max - self.b_min)/precision), 4))

    def get_knn(self, x, y):
        """
        :param x: (float) sensor_a coordinate of the point
        :param y: (float) sensor_b coordinate of the point
        :return: the indexes of the K nearest neighbours
        """
        distances = (self.a-x)**2 + (self.b-y)**2
        return distances.argsort()[:self.k]

    def get_probas(self, x, y):
        """
        :param x: (float) sensor_a coordinate of the point
        :param y: (float) sensor_b coordinate of the point
        :return: (numpy array) the score for each category
        """
        knn = self.get_knn(x, y)
        labels = self.labels[knn]

        proba_0 = np.mean(labels == 0)
        proba_1 = np.mean(labels == 1)
        proba_2 = np.mean(labels == 2)
        proba_3 = np.mean(labels == 3)

        return [proba_0, proba_1, proba_2, proba_3]

    def predict(self, a, b):
        """
        :param a: (numpy array) sensor_a data to predict
        :param b: (numpy array) sensor_b data to predict
        :return: (numpy array) the score for each category for each example
        """
        assert len(a) == len(b)
        answer = np.zeros((len(a), 4))
        for u in range(len(a)):
            answer[u, :] = self.get_probas(a[u], b[u])
        return answer

    # def train(self):
    #     print("training...")
    #     for i in range(len(self.grid[:, 0, 0])):
    #         print(i)
    #         for j in range(len(self.grid[0, :, 0])):
    #             self.grid[i, j, :] = self.get_probas(i * self.precision + self.a_min,
    #                                                  j * self.precision + self.b_min)

    # def predict(self, a, b):
    #     assert len(a) == len(b)
    #     answer = np.zeros((len(a), 4))
    #     for u in range(len(a)):
    #
    #         i = int((a[u] - self.a_min)/self.precision)
    #         j = int((b[u] - self.b_min)/self.precision)
    #         i = min(self.a_max, i)
    #         i = max(self.a_min, i)
    #         j = min(self.b_max, j)
    #         j = max(self.b_min, j)
    #         answer[u, :] = self.grid[i, j, :]
    #
    #     return answer


