import numpy as np
import math
from TwoDimensionalClassification.madge_calculator import gaussian_area, euclidean


class ClassificationSet(object):
    def __init__(self, mean=0, sigma=1):
        self.graph = {}
        self.vectorized_graph = []
        self.mean = mean  # We will most likely hold this at 0 since we're always starting from the point
        self.sigma = sigma
        self.range_vector = None  # each element is the w_i, sum(n) is calculated. The range vector should be the dim of the set.

    @property
    def show(self):
        for point in self.graph:
            print(point)
        return True

    def add_point(self, point):
        """
        Adds another point to the set. The set will be an undirected graph with nC2 total edges
        :param point: a Point object
        :return: None
        """
        if self.graph.get(point):
            print('Point exists in set')
        else:
            self.graph[point] = {}
            # Add point to np array
            self.vectorized_graph = np.append(self.vectorized_graph, point)

    def point_weight_vectorize(self, self_graph_point, input_point):
        """
        Helper function for creating zi_wi for 2 points
        :param self_graph_point: a Point object
        :param input_point: a Point object
        :return: (zi*wi, wi) tuple
        """
        # We could probably optimize here with some array x array apply methods with numpy
        # TODO: we need a relative distance here, because otherwise we're gonna have massive distances between two points in space
        distance = euclidean(input_point.tuple, self_graph_point.tuple)
        w_i = gaussian_area(distance, self.mean, self.sigma)
        z_i = self_graph_point.type
        return np.multiply(w_i, z_i), w_i

    def calculate_madge_data_and_map_to_point(self, point, sigma):
        """
        Gives the weight of a point given the current set of points. This will just use eucliean distance with a sigma of 1
        :param point: n-dim point
        :param normalize: sigma calculation
        :return: weight as a reshaped vector
        """
        #

        sum_zi_wi = 0
        sum_wi = 0
        self.sigma = sigma
        for classification_point in self.vectorized_graph:
            weight_wi_zi, weight_xi = self.point_weight_vectorize(classification_point, point)
            sum_zi_wi = np.add(sum_zi_wi, weight_wi_zi)
            sum_wi = np.add(sum_wi, weight_xi)
        if sum_wi == 0 or math.isnan(np.divide(sum_zi_wi, sum_wi)):
            # this NAN default isn't necessarily good for classification without a default
            return 0
        return np.divide(sum_zi_wi, sum_wi)
