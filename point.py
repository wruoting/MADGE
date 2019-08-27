class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __repr__(self):
        return '({}, {})'.format(self.x, self.y)
    
    @property
    def tuple(self):
        return (self.x, self.y)