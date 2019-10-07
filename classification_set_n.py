from classification_set import ClassificationSet
import numpy as np
import math


class ClassificationSetN(ClassificationSet):

    def __init__(self, mean=0, sigma=1):
        super().__init__(mean=mean, sigma=sigma)

    def calculate_madge_data_and_map_to_point(self, point):
        """
        Gives the weight of a point given the current set of points
        :param point: n-dim point 
        :return: weight as a reshaped vector
        """
        # Recalculate sigma with f(x)
        for classification_point in self.vectorized_graph:
            # each classification point will need to be distanced by dimension to this point, with an input of []
            distance_between_points_vector = np.absolute(np.subtract(classification_point.tuple, point.tuple))
            sum_distance_between_points = np.sum(distance_between_points_vector)
            normalized_vector = np.divide(np.absolute(np.subtract(classification_point.tuple, point.tuple)), sum_distance_between_points)
            # we multiply each vector by its corresponding range_vector
            self.sigma = np.dot(normalized_vector, self.range_vector)

        sum_zi_wi = 0
        sum_wi = 0
        vectorized_point_weight_function = np.vectorize(self.point_weight_vectorize)
        for data_point in self.vectorized_graph:
            weight_wi_zi, weight_xi = vectorized_point_weight_function(data_point, point)
            sum_zi_wi = np.add(sum_zi_wi, weight_wi_zi)
            sum_wi = np.add(sum_wi, weight_xi)
        if math.isnan(np.divide(sum_zi_wi, sum_wi)):
            return 0
        return np.divide(sum_zi_wi, sum_wi)
