from classification_set import ClassificationSet
from point import Point
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

new_set = ClassificationSet()

new_set.add_point(Point(1,2,1))
new_set.add_point(Point(1,3,1))
new_set.add_point(Point(0,0,0))

# https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)

X, Y = np.meshgrid(x, y)
Z = new_set.weight(X, Y)
# 
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');
ax.view_init(60, 35)
plt.show()
# new_set.show