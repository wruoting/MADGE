from point import Point
from madge_calculator import gaussian_area


class Set(object):
    def __init__(self):
        pass

    @property
    def undirected_graph(self):
        """
        Returns the set of undirected graph points as an array of tuples
        :return:
        """
        return

    def search_by_tuple(self):
        """
        Allows retrieval of the distance between a pair of points given a tuple
        :return: Distance between two points as a float
        """
        return

    def weight(self, point):
        """
        Gives the weight of a point given the current set of points
        :return:
        """
        return

    def add_point(self, point):
        """
        Adds another point to the set. The set will be an undirected graph with nC2 total edges
        :param point: a Point object
        :return: None
        """
        pass
