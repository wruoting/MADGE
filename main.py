from classification_set import ClassificationSet
from point import Point
from mpl_toolkits import mplot3d
from matplotlib import cm
import numpy as np
import plotly as py
from plotly.offline import init_notebook_mode
import plotly.graph_objs as go
import plotly.graph_objects as go
import ast
import pandas as pd


def add_points_and_graph(data_set, linspace, filename):
    """
    Gives the weight of a point given the current set of points
    :param data_set: set of tuples [(a,b, classification), (), ()] in an array
    :param linspace: linspace as a vector (-x, x, total_space)
    :return: weight as a reshaped vector
    """
    # TODO: get your classifications to accept the whole list of unique classifiers
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
        if data_point[2] == 1:
            x_1.append(data_point[0])
            y_1.append(data_point[1])
            z_1.append(0)
        else:
            x_0.append(data_point[0])
            y_0.append(data_point[1])
            z_0.append(0)

    # https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
    x_space = np.linspace(linspace[0], linspace[1], linspace[2])
    y_space = np.linspace(linspace[0], linspace[1], linspace[2])

    X, Y = np.meshgrid(x_space, y_space)
    Z = new_set.weight(X, Y)
    trace_surface = go.Surface(x=X, y=Y, z=Z)
    trace_scatter_class_a = go.Scatter3d(x=x_1, y=y_1, z=z_1, mode='markers')
    trace_scatter_class_b = go.Scatter3d(x=x_0, y=y_0, z=z_0, mode='markers')
    data = [trace_surface, trace_scatter_class_a, trace_scatter_class_b]
    fig = go.Figure(data=data)
    fig.update_layout(title='MADGE Graph', autosize=False,
                  width=700, height=700,
                  margin=dict(l=65, r=50, b=65, t=90))
    fig.show() 
    py.offline.plot(fig,filename=filename)             

def add_points_to_data_set():
    f= open("./SampleData/circle_classification.txt","r")
    return np.array(ast.literal_eval(f.readlines()[0]))

def convert_array_to_array_of_tuples(input_array):
    to_tuple = lambda v: tuple(v)
    return [to_tuple(ai) for ai in input_array]

# data_set = [(5,5,1), (1,1,1), (0,0,0), (-5,-5,1), (-3,-3,1)]

linspace = (-6, 6, 30)
training_data_set = convert_array_to_array_of_tuples(add_points_to_data_set())
filename = '09-09-2019-TestDataSet-MADGE'
add_points_and_graph(training_data_set, linspace, filename)
# add_points_and_graph(data_set, linspace)



