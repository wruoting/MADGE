from TwoDimensionalClassification.point import Point
from madge_calculator import gaussian_area, euclidean
import numpy as np


class ClassificationSet(object):
    def __init__(self, mean=0, sigma=1):
        self.graph = {}
        self.vectorized_graph = []
        self.mean = mean  # We will most likely hold this at 0 since we're always starting from the point
        self.sigma = sigma

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
        distance = euclidean(input_point.tuple, self_graph_point.tuple)
        w_i = gaussian_area(distance, self.mean, self.sigma)
        z_i = self_graph_point.type
        return np.multiply(w_i, z_i), w_i

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
