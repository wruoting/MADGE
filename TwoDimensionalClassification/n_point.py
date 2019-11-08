class NPoint(object):
    def __init__(self, point_tuple, type=None):
        self.point_tuple = point_tuple
        self.type = type

    @property
    def tuple(self):
        return self.point_tuple
