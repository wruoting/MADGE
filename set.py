from point import Point
from madge_calculator import gaussian_area, euclidean
import numpy as np

class Set(object):
    def __init__(self, mean=0, sigma=1):
        self.graph = {}
        self.vectorized_graph = []
        self.two_dimensional_vectorized_graph = None
        self.mean = mean # We will most likely hold this at 0 since we're always starting from the point
        self.sigma = sigma
        
    @property
    def show(self):
        for point in self.graph:
            print(point)
            
            
    def weight(self, x, y):
        """
        Gives the weight of a point given the current set of points
        :param x: x coordinate np array like
        :param y: y coordinate np array like
        :return: weight as a float
        """
        np_point_array = self.vectorized_points(x, y)
        point_weight_vectorize = np.vectorize(self.point_to_point_weight)
        print(np_point_array)
        print(self.two_dimensional_vectorized_graph)
        [val, val2] = point_weight_vectorize(np_point_array, self.two_dimensional_vectorized_graph)
        print(self.vectorized_graph)
        print(self.two_dimensional_vectorized_graph)
        print(np.ndim(self.two_dimensional_vectorized_graph))
        # point_weight_vectorize(np_point_array, self.vectorized_graph)
        # return np.divide(sigma_w_z, sigma_w)
        
    def add_point(self, point):
        """
        Adds another point to the set. The set will be an undirected graph with nC2 total edges
        :param point: a Point object
        :param type: an int classifier
        :return: None
        """
        if self.graph.get(point):
            print('Point exists in set')
        else:
            self.graph[point] = {}
            # Add point to np array
            self.vectorized_graph = np.append(self.vectorized_graph, point)
            # Make into a 2x2 d array
            self.two_dimensional_vectorized_graph = np.tile(self.vectorized_graph, (len(self.vectorized_graph), 1))
            
    def point_to_point_weight(self, point, graph_point):
        # We could probably optimize here with some array x array apply methods with numpy
        distance = euclidean(point.tuple, graph_point.tuple)
        w_i = gaussian_area(distance, self.mean, self.sigma)
        z_i = graph_point.type
        return [np.multiply(w_i, z_i), w_i]
            
    @staticmethod
    def vectorized_points(x, y):
        def to_point(x_in, y_in):
            return Point(x_in,y_in)
        point_func = np.vectorize(to_point)
        return point_func(x, y)   
    
    

        
