from classification_set import ClassificationSet
import numpy as np
import math

from madge_calculator import gaussian_area


class ClassificationSetN(ClassificationSet):

    def __init__(self, mean=0, sigma=1):
        super().__init__(mean=mean, sigma=sigma)
        self.normalized_range_vector = None
        self.normalized_gaussian_vector = None

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
                # Do a gaussian on the range with the normalization factor as the division of the range
                self.normalized_range_vector = np.divide(self.range_vector, self.normalization_standard_deviation_factor)
                # Now we normalize this range vector with our input point as our mean
                gauss_vectorize = np.vectorize(gaussian_area)
                self.normalized_gaussian_vector = gauss_vectorize(classification_point.tuple, point.tuple, self.normalized_range_vector)
                self.sigma = np.dot(normalized_vector, self.normalized_gaussian_vector)
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
            # this NAN default isn't necessarily good for classification without a default
            return 0
        return np.round(np.divide(sum_zi_wi, sum_wi))

    def calculate_madge_data_and_map_to_point_v2(self, point, sigma):
        """
        Gives the weight of a point given the current set of points
        :param point: n-dim point
        :param normalize: sigma calculation
        :return: weight as a reshaped vector
        """
        sum_zi_wi = 0
        for classification_point in self.vectorized_graph:
            self.sigma = sigma
            vectorized_point_weight_function = np.vectorize(self.point_weight_vectorize_range_weighted)
            weight_wi_zi = vectorized_point_weight_function(classification_point, point)
            sum_zi_wi = np.add(sum_zi_wi, weight_wi_zi)
        return sum_zi_wi

    def calculate_madge_data_and_map_to_point_v3(self, point, sigma):
        """
        Gives the weight of a point given the current set of points
        :param point: n-dim point
        :param normalize: sigma calculation
        :return: weight as a reshaped vector
        """
        sum_zi_wi = 0
        for classification_point in self.vectorized_graph:
            self.sigma_factor_vector = sigma
            vectorized_point_weight_function = np.vectorize(self.point_weight_vectorize_range_sigma_weighted)
            weight_wi_zi = vectorized_point_weight_function(classification_point, point)
            sum_zi_wi = np.add(sum_zi_wi, weight_wi_zi)
        return sum_zi_wi

    def calculate_madge_data_and_map_to_point_v4(self, point, sigma):
        """
        Gives the weight of a point given the current set of points
        :param point: n-dim point
        :param normalize: sigma calculation
        :return: weight as a reshaped vector
        """
        #
        sum_zi_wi = 0
        for classification_point in self.vectorized_graph:
            self.sigma = sigma
            # vectorized_point_weight_function = np.vectorize(self.point_weight_vectorize_gaussian_weighted)
            weight_wi_zi = self.point_weight_vectorize_gaussian_weighted(classification_point, point)
            sum_zi_wi = np.add(sum_zi_wi, weight_wi_zi)
        print('sum_zi_wi', sum_zi_wi)
        return sum_zi_wi

