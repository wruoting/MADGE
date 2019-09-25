from classification_set import ClassificationSet
import numpy as np


class ClassificationSetN(ClassificationSet):

    def __init__(self, mean=0, sigma=1):
        super().__init__(mean=mean, sigma=sigma)

    def calculate_madge_data_and_map_to_point(self, point):
        """
        Gives the weight of a point given the current set of points
        :param point: n-dim point 
        :return: weight as a reshaped vector
        """
        sum_zi_wi = 0
        sum_wi = 0
        vectorized_point_weight_function = np.vectorize(self.point_weight_vectorize)
        for data_point in self.vectorized_graph:
            weight_wi_zi, weight_xi = vectorized_point_weight_function(data_point, point)
            sum_zi_wi = np.add(sum_zi_wi, weight_wi_zi)
            sum_wi = np.add(sum_wi, weight_xi)

        return np.divide(sum_zi_wi, sum_wi)
