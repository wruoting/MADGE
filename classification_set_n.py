from classification_set import ClassificationSet
import numpy as np
import math


class ClassificationSetN(ClassificationSet):

    def __init__(self, mean=0, sigma=1):
        super().__init__(mean=mean, sigma=sigma)

    def calculate_madge_data_and_map_to_point(self, point, normalize=True):
        """
        Gives the weight of a point given the current set of points
        :param point: n-dim point
        :param normalize: sigma calculation
        :return: weight as a reshaped vector
        """
        sum_zi_wi = 0
        sum_wi = 0
        # Recalculate sigma with f(x)
        if normalize:
            for classification_point in self.vectorized_graph:
                self.sigma = 0
                # each classification point will need to be distanced by dimension to this point
                # a specific sigma will be calculated per point based on the weights of the other points
                distance_between_points_vector = np.absolute(np.subtract(classification_point.tuple, point.tuple))
                sum_distance_between_points = np.sum(distance_between_points_vector)
                normalized_vector = np.divide(np.absolute(np.subtract(classification_point.tuple, point.tuple)), sum_distance_between_points)
                self.sigma = np.dot(normalized_vector, self.range_vector)
                # From this point on, we are calculating per point, it just so happens that if we are not normalizing,
                # we will use just one sigma
                vectorized_point_weight_function = np.vectorize(self.point_weight_vectorize)
                weight_wi_zi, weight_xi = vectorized_point_weight_function(classification_point, point)
                sum_zi_wi = np.add(sum_zi_wi, weight_wi_zi)
                sum_wi = np.add(sum_wi, weight_xi)
        else:
            vectorized_point_weight_function = np.vectorize(self.point_weight_vectorize)
            for data_point in self.vectorized_graph:
                weight_wi_zi, weight_xi = vectorized_point_weight_function(data_point, point)
                sum_zi_wi = np.add(sum_zi_wi, weight_wi_zi)
                sum_wi = np.add(sum_wi, weight_xi)
        if math.isnan(np.divide(sum_zi_wi, sum_wi)):
            return 0
        return np.divide(sum_zi_wi, sum_wi)
