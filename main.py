from set import Set
from point import Point
from mpl_toolkits import mplot3d
import numpy as np

new_set = Set()

new_set.add_point(Point(1,2,0))
new_set.add_point(Point(1,3,0))
new_set.add_point(Point(0,0,0))

x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)

X, Y = np.meshgrid(x, y)
Z = new_set.weight(X, Y)
# 
# fig = plt.figure()
# ax = plt.axes(project='3d')
# ax.contour3D()

new_set.show