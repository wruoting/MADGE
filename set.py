from point import Point
from madge_calculator import gaussian_area, euclidean


class Set(object):
    def __init__(self):
        self.graph = {}
    
    @property
    def show(self):
        index = 0
        for point in self.graph:
            print(point)
            self.__show(self.graph[point], index)
        return
    
    def __show(self, graph, index):
        if graph is not None:
            index += 2
            for point in graph:
                if isinstance(graph[point], Point):
                    print('{}{}'.format(' '*index, point))
                    self.__show(graph[point], index)
                else:
                    print('{}{} : {}'.format(' '*index, point, graph[point]))
    
    @property
    def undirected_graph(self):
        """
        Returns the set of undirected graph points as an array of tuples
        :return:
        """
        return
    
    def search_by_tuple(self, point_a, point_b):
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
        if self.graph.get(point):
            print('Point exists in set')
        else:
            # Connect to existing points if exists, then add to graph
            # Looking up tuples will look for both pairs
            for index_point in self.graph:
                self.graph[index_point].update({point: euclidean(index_point.tuple, point.tuple)})
            self.graph[point] = {}
                    
           
            
