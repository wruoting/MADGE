from classification_set import ClassificationSet
from point import Point
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import ast

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
    classification_a = 1
    classification_b = 0
    for data_point in data_set:
        new_set.add_point(Point(data_point[0], data_point[1], data_point[2]))
        if data_point[2] != 1:
            classification_a = data_point[2]
        if data_point[2] != classification_a:
            classification_b = data_point[2]
        if data_point[2] == classification_a:
            x_1.append(data_point[0])
            y_1.append(data_point[1])
            z_1.append(0)
        else:
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
    ax.scatter(x_1, y_1, z_1, c="#ff0000", label=classification_a)
    # The blue is the 0 classification
    ax.scatter(x_0, y_0, z_0, c="#003399", label=classification_b)
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

def add_points_to_data_set():
    f= open("./SampleData/circle_classification.txt","r")
    return np.array(ast.literal_eval(f.readlines()[0]))

def convert_array_to_array_of_tuples(input_array):
    to_tuple = lambda v: tuple(v)
    return [to_tuple(ai) for ai in input_array]

# data_set = [(5,5,1), (1,1,1), (0,0,0), (-5,-5,1), (-3,-3,1)]

linspace = (-6, 6, 30)
training_data_set = convert_array_to_array_of_tuples(add_points_to_data_set())
add_points_and_graph(training_data_set, linspace)

# add_points_and_graph(data_set, linspace)



