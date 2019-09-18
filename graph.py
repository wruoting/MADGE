from classification_set import ClassificationSet
from point import Point
import numpy as np
import plotly as py
import plotly.graph_objects as go
import ast


def create_data_graphs_and_classifiers(data_set, linspace, classifiers, surface_name='MADGE Surface'):
    """
    Compiles and converts data set into an array tht can be graphed
    :param data_set: set of tuples [(a,b, classification), (), ()] in an array
    :param linspace: linspace as a vector (-x, x, total_space)
    :param classifiers: classifiers as a tuple to indicate what the classification is eg. (0, 1) is for two way classification
    :param surface_name: the title of the graph for the surface
    :return: data in the form of an array with Surface and Scatter3D objects [Surface, Scatter3D, Scatter3D]
    """
    new_set = ClassificationSet(sigma=0.1)
    x_1 = []
    y_1 = []
    z_1 = []
    x_0 = []
    y_0 = []
    z_0 = []

    for data_point in data_set:
        new_set.add_point(Point(data_point[0], data_point[1], data_point[2]))
        if data_point[2] == classifiers[0]:
            x_1.append(data_point[0])
            y_1.append(data_point[1])
            z_1.append(0)
        elif data_point[2] == classifiers[1]:
            x_0.append(data_point[0])
            y_0.append(data_point[1])
            z_0.append(0)

    # https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
    x_space = np.linspace(linspace[0], linspace[1], linspace[2])
    y_space = np.linspace(linspace[0], linspace[1], linspace[2])

    X, Y = np.meshgrid(x_space, y_space)
    Z = new_set.calculate_madge_data_and_map_to_plane(X, Y)
    trace_surface = go.Surface(x=X, y=Y, z=Z, name=surface_name)
    trace_scatter_class_a = go.Scatter3d(x=x_1, y=y_1, z=z_1, mode='markers',
                                         name='Classifier {}'.format(classifiers[0]))
    trace_scatter_class_b = go.Scatter3d(x=x_0, y=y_0, z=z_0, mode='markers',
                                         name='Classifier {}'.format(classifiers[1]))
    data = [trace_surface, trace_scatter_class_a, trace_scatter_class_b]
    return data
    
    
def create_and_plot_points_in_data_set(data_set, classifiers):
    """
    Creates and returns a data object for just the scatter plot with no third graph
    :param data_set: set of tuples [(a,b, classification), (), ()] in an array
    :param classifiers: classifiers as a tuple to indicate what the classification is eg. (0, 1) is for two way classification
    :return: data in the form of an array with Scatter3D objects [Scatter3D, Scatter3D]
    """
    x_1 = []
    y_1 = []
    z_1 = []
    x_0 = []
    y_0 = []
    z_0 = []

    for data_point in data_set:
        if data_point[2] == classifiers[0]:
            x_1.append(data_point[0])
            y_1.append(data_point[1])
            z_1.append(0)
        elif data_point[2] == classifiers[1]:
            x_0.append(data_point[0])
            y_0.append(data_point[1])
            z_0.append(0)

    trace_scatter_class_a = go.Scatter3d(x=x_1, y=y_1, z=z_1, mode='markers',
                                         name='Classifier {}'.format(classifiers[0]))
    trace_scatter_class_b = go.Scatter3d(x=x_0, y=y_0, z=z_0, mode='markers',
                                         name='Classifier {}'.format(classifiers[1]))
    data = [trace_scatter_class_a, trace_scatter_class_b]
    return data
    
    
def read_data_from_file(path):
    """
    Read data from a file and returns it as a numpy array
    :param path: path of file relative to this file
    :return: numpy array
    """
    f = open(path, "r")
    return np.array(ast.literal_eval(f.readlines()[0]))


def convert_array_to_array_of_tuples(input_array):
    """
    Converts each array of an array into tuples
    :param input_array: path of file relative to this file
    :return: array of tuples
    """
    to_tuple = lambda v: tuple(v)
    return [to_tuple(ai) for ai in input_array]


def plot_madge_data(data, filename, title):
    """
    Generates a plot of the madge data 
    :param data: data in the form of go scatter/3d data
    :param filename: name of the output file (will append .html to the end if it is not included)
    :param title: title of the graph
    :return: None
    """
    fig = go.Figure(data=data)
    fig.update_layout(title=title, autosize=True,
                      width=700, height=700,
                      margin=dict(l=65, r=50, b=65, t=90))
    py.offline.plot(fig, filename=filename)


def plot_scatter_data(data, filename, title):
    """
    Generates a plot of the madge data with only the scatter
    :param data: data in the form of go scatter/3d data
    :param filename: name of the output file (will append .html to the end if it is not included)
    :param title: title of the graph
    :return: None
    """
    fig = go.Figure(data=data)
    fig.update_layout(title=title)
    py.offline.plot(fig, filename=filename)
