from classification_set import ClassificationSet
from point import Point
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt


def add_points_and_graph(data_set, linspace):
    """
    Gives the weight of a point given the current set of points
    :param data_set: set of tuples [(a,b, classification), (), ()] in an array
    :param linspace: linspace as a vector (-x, x, total_space)
    :return: weight as a reshaped vector
    """
    new_set = ClassificationSet()
    x_1 = []
    y_1 = []
    z_1 = []
    x_0 = []
    y_0 = []
    z_0 = []
    for data_point in data_set:
        new_set.add_point(Point(data_point[0], data_point[1], data_point[2]))
        if data_point[2] == 1:
            x_1.append(data_point[0])
            y_1.append(data_point[1])
            z_1.append(0)
        elif data_point[2] == 0:
            x_0.append(data_point[0])
            y_0.append(data_point[0])
            z_0.append(0)

    fig = plt.figure()
    # https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
    x_space = np.linspace(linspace[0], linspace[1], linspace[2])
    y_space = np.linspace(linspace[0], linspace[1], linspace[2])

    X, Y = np.meshgrid(x_space, y_space)
    Z = new_set.weight(X, Y)

    ax = plt.axes(projection='3d')
    # The red is the 1 classification
    ax.scatter(x_1, y_1, z_1, c="#ff0000", label="1")
    # The blue is the 0 classification
    ax.scatter(x_0, y_0, z_0, c="#003399", label="0")
    # For contouring
    # ax.contour3D(X, Y, Z, 50, cmap='binary')
    # For surface plots
    ax.plot_surface(X, Y, Z)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z');
    ax.view_init(60, 35)
    ax.legend()
    plt.show()

data_set = [(1,2,1), (1,3,1), (0,0,0)]
linspace = (-6, 6, 30)
add_points_and_graph(data_set, linspace)