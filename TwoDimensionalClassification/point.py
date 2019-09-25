class Point(object):
    def __init__(self, x, y, type=None):
        self.x = x
        self.y = y
        self.type = type
        
    def __repr__(self):
        return '({}, {}, {})'.format(self.x, self.y, self.type)
    
    @property
    def tuple(self):
        return self.x, self.y