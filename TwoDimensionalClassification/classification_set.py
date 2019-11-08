from TwoDimensionalClassification.point import Point
from TwoDimensionalClassification.madge_calculator import gaussian_area, euclidean
import numpy as np


class ClassificationSet(object):
    def __init__(self, mean=0, sigma=1):
        self.graph = {}
        self.vectorized_graph = []
        self.mean = mean  # We will most likely hold this at 0 since we're always starting from the point
        self.sigma = sigma
        # We are going to use the equation sum(n_i/sum(n) * range(w_i)/6),
        # where n_i is the ith dimension of the point we are classifying
        self.range_vector = None  # each element is the w_i, sum(n) is calculated. The range vector should be the dim of the set.
        self.normalization_standard_deviation_factor = 0.1545  # the number of standard deviations we are normalizing by
        self.sigma_factor_vector = None  # The factor for the sigmas eg. 0.1 to 6 for each dimension

    @property
    def show(self):
        for point in self.graph:
            print(point)

    def calculate_madge_data_and_map_to_plane(self, x, y):
        """
        Gives the weight of a point given the current set of points
        :param x: x coordinate np array like
        :param y: y coordinate np array like
        :return: weight as a reshaped vector
        """
        # the input points to graph
        np_point_array = self.vectorized_points(x, y).reshape(len(x) * len(y))
        # nxn based on np point array to map to for our function
        # it HAS to be a square matrix!!
        # the points in our system
        vectorized_graph_array = np.tile(self.vectorized_graph, (np.shape(np_point_array)[0], 1))
        # Transpose and then turn into [ p1 p1 p1 ... p1] [p2 p2 p2 ... p2] etc. n times
        vectorized_graph_array_trans = vectorized_graph_array.transpose()
        np.shape(vectorized_graph_array_trans)

        ## Create our vectorize function
        vectorized_point_weight_function = np.vectorize(self.point_weight_vectorize)
        # for each point in our system, apply it to all points
        sum_zi_wi = np.zeros(np.shape(vectorized_graph_array_trans)[1])
        sum_wi = np.zeros(np.shape(vectorized_graph_array_trans)[1])
        # this generates the weight times zi and the weight for each input point to point ratio
        for index, data_point_row in enumerate(vectorized_graph_array_trans):
            weight_wi_zi, weight_xi = vectorized_point_weight_function(data_point_row, np_point_array)
            # all of the even indices are wi/zi and all the odd indices are wi
            # add all even indices together and all the odd indices together as well
            sum_zi_wi = np.add(sum_zi_wi, weight_wi_zi)
            sum_wi = np.add(sum_wi, weight_xi)

        madge_vector = np.nan_to_num(np.divide(sum_zi_wi, sum_wi))
        # need to reshape this madge vector
        return madge_vector.reshape(len(x), len(y))

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

    def point_weight_vectorize_range_weighted(self, self_graph_point, input_point):
        """
        Helper function for creating zi_wi for 2 points. This method uses the range vector of our equation
        to map weighted points instead
        :param self_graph_point: a Point object
        :param input_point: a Point object
        :return: (zi*wi, wi) tuple
        """

        # We individually calculate weights here with the inner range first
        inner_range = np.absolute(np.subtract(input_point.tuple, self_graph_point.tuple))
        inner_range_sum = np.sum(inner_range)
        inner_range_vector = np.divide(inner_range, inner_range_sum)
        # Calculate the effect of range on sigma
        sigma_range_vector = np.divide(self.range_vector, self.sigma)
        # Create a vectorized gaussian using the outer range
        gaussian_vectorized = np.vectorize(gaussian_area)
        if self.range_vector is None:
            raise Exception("You need a range vector to do this calculation")
        inner_gaussian = gaussian_vectorized(self_graph_point.tuple, input_point.tuple, sigma_range_vector)
        weighted_gaussian = np.dot(inner_range_vector, inner_gaussian)
        # L2 norm this vector
        # weighted_gaussian_norm = norm(weighted_gaussian_vector)

        z_i = self_graph_point.type
        return np.multiply(weighted_gaussian, z_i)

    def point_weight_vectorize_range_sigma_weighted(self, self_graph_point, input_point):
        """
        Helper function for creating zi_wi for 2 points. This method uses an optimal sigma valuation based on the range
        vector rather than a divisor method
        :param self_graph_point: a Point object
        :param input_point: a Point object
        :return: (zi*wi, wi) tuple
        """

        # We individually calculate weights here with the inner range first
        inner_range = np.absolute(np.subtract(input_point.tuple, self_graph_point.tuple))
        inner_range_sum = np.sum(inner_range)
        inner_range_vector = np.divide(inner_range, inner_range_sum)
        # Create a vectorized gaussian using the outer range
        gaussian_vectorized = np.vectorize(gaussian_area)
        if self.range_vector is None:
            raise Exception("You need a range vector to do this calculation")
        # this is the calculation for the sigma to range
        sigma_final = np.dot(self.range_vector, self.sigma_factor_vector)
        inner_gaussian = gaussian_vectorized(self_graph_point.tuple, input_point.tuple, sigma_final)
        weighted_gaussian = np.dot(inner_range_vector, inner_gaussian)
        z_i = self_graph_point.type
        return np.multiply(weighted_gaussian, z_i)

    def point_weight_vectorize_gaussian_weighted(self, self_graph_point, input_point):
        """
        Helper function for creating zi_wi for 2 points. This method uses a gaussian weighted for each dimension and then euclidean distances the vector of weights
        :param self_graph_point: a Point object
        :param input_point: a Point object
        :return: weighted gaussian tuple (g1, g2...gn)
        """
        # Create a vectorized gaussian using the outer range
        gaussian_vectorized = np.vectorize(gaussian_area)
        if self.range_vector is None:
            raise Exception("You need a range vector to do this calculation")
        # Calculate the effect of range on sigma
        sigma_range_vector = np.divide(self.range_vector, self.sigma)
        inner_gaussian = gaussian_vectorized(self_graph_point.tuple, input_point.tuple, sigma_range_vector)
        return np.multiply(np.linalg.norm(inner_gaussian), self_graph_point.type)

    @staticmethod
    def vectorized_points(x, y):
        """
        Adds another point to the set. The set will be an undirected graph with nC2 total edges
        :param x: int x
        :param y: int y
        :return: vectorized function
        """

        def to_point(x_in, y_in):
            return Point(x_in, y_in)

        point_func = np.vectorize(to_point)
        return point_func(x, y)
