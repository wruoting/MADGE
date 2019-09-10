from classification_set import ClassificationSet
from point import Point
import numpy as np
import plotly as py
from plotly.offline import init_notebook_mode
import plotly.graph_objs as go
import plotly.graph_objects as go
import ast
import pandas as pd


def add_points_and_graph(data_set, linspace, classifiers):
    """
    Gives the weight of a point given the current set of points
    :param data_set: set of tuples [(a,b, classification), (), ()] in an array
    :param linspace: linspace as a vector (-x, x, total_space)
    :param classifiers: classifiers as a tuple to indicate what the classification is
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
    Z = new_set.weight(X, Y)
    trace_surface = go.Surface(x=X, y=Y, z=Z, name='MADGE Surface')
    trace_scatter_class_a = go.Scatter3d(x=x_1, y=y_1, z=z_1, mode='markers', name='Classifier {}'.format(classifiers[0]))
    trace_scatter_class_b = go.Scatter3d(x=x_0, y=y_0, z=z_0, mode='markers', name='Classifier {}'.format(classifiers[1]))
    data = [trace_surface, trace_scatter_class_a, trace_scatter_class_b]
    return data

def create_and_plot_points_in_data_set(data_set, filename, classifiers, title):
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
                 
    trace_scatter_class_a = go.Scatter3d(x=x_1, y=y_1, z=z_1, mode='markers', name='Classifier {}'.format(classifiers[0]))
    trace_scatter_class_b = go.Scatter3d(x=x_0, y=y_0, z=z_0, mode='markers', name='Classifier {}'.format(classifiers[1]))
    data = [trace_scatter_class_a, trace_scatter_class_b]
    fig = go.Figure(data=data)
    fig.update_layout(title=title)
    py.offline.plot(fig,filename=filename)  
    
def add_points_to_data_set(path):
    f= open(path,"r")
    return np.array(ast.literal_eval(f.readlines()[0]))

def convert_array_to_array_of_tuples(input_array):
    to_tuple = lambda v: tuple(v)
    return [to_tuple(ai) for ai in input_array]

def plot_madge_data(data, filename, title):
    fig = go.Figure(data=data)
    fig.update_layout(title=title, autosize=True,
                  width=700, height=700,
                  margin=dict(l=65, r=50, b=65, t=90))
    py.offline.plot(fig,filename=filename)    
# data_set = [(5,5,1), (1,1,1), (0,0,0), (-5,-5,1), (-3,-3,1)]

linspace = (-6, 6, 30)
path = "./SampleData/ClassifyCircleData.txt"
training_data_set = convert_array_to_array_of_tuples(add_points_to_data_set(path))
training_data_set_filename = '09-09-2019-TestDataSet-2D-Circle-Classification'
filename = '09-09-2019-TestDataSet-MADGE-Circle-Classification'
data = add_points_and_graph(training_data_set, linspace, (1, -1))
title = 'Circle Classification'
create_and_plot_points_in_data_set(training_data_set, training_data_set_filename, (1, -1), title)
plot_madge_data(data, filename, title)


linspace = (-6, 6, 30)
path = "./SampleData/ClassifyXORData.txt"
training_data_set = convert_array_to_array_of_tuples(add_points_to_data_set(path))
training_data_set_filename = '09-09-2019-TestDataSet-2D-ClassifyXORData'
filename = '09-09-2019-TestDataSet-MADGE-ClassifyXORData'
data = add_points_and_graph(training_data_set, linspace, (1, -1))
title = 'Classify XOR Data'
create_and_plot_points_in_data_set(training_data_set, training_data_set_filename, (1, -1), title)
plot_madge_data(data, filename, title)

linspace = (-6, 6, 30)
path = "./SampleData/ClassifyTwoGaussData.txt"
training_data_set = convert_array_to_array_of_tuples(add_points_to_data_set(path))
training_data_set_filename = '09-09-2019-TestDataSet-2D-ClassifyTwoGaussData'
filename = '09-09-2019-TestDataSet-MADGE-ClassifyTwoGaussData'
data = add_points_and_graph(training_data_set, linspace, (1, -1))
title = 'Classify Two Gauss Data'
create_and_plot_points_in_data_set(training_data_set, training_data_set_filename, (1, -1), title)
plot_madge_data(data, filename, title)

linspace = (-6, 6, 30)
path = "./SampleData/ClassifySpiralData.txt"
training_data_set = convert_array_to_array_of_tuples(add_points_to_data_set(path))
training_data_set_filename = '09-09-2019-TestDataSet-2D-ClassifySpiralData'
filename = '09-09-2019-TestDataSet-MADGE-ClassifySpiralData'
data = add_points_and_graph(training_data_set, linspace, (1, -1))
title = 'Classify Spiral Data'
create_and_plot_points_in_data_set(training_data_set, training_data_set_filename, (1, -1), title)
plot_madge_data(data, filename, title)

linspace = (-6, 6, 30)
path = "./SampleData/RegressPlane.txt"
training_data_set = convert_array_to_array_of_tuples(add_points_to_data_set(path))
training_data_set_filename = '09-09-2019-TestDataSet-2D-RegressPlane'
filename = '09-09-2019-TestDataSet-MADGE-RegressPlane'
data = add_points_and_graph(training_data_set, linspace, (1, -1))
title = 'Regress Plane'
create_and_plot_points_in_data_set(training_data_set, training_data_set_filename, (1, -1), title)
plot_madge_data(data, filename, title)

linspace = (-6, 6, 30)
path = "./SampleData/RegressGaussian.txt"
training_data_set = convert_array_to_array_of_tuples(add_points_to_data_set(path))
training_data_set_filename = '09-09-2019-TestDataSet-2D-RegressGaussian'
filename = '09-09-2019-TestDataSet-MADGE-RegressGaussian'
data = add_points_and_graph(training_data_set, linspace, (1, -1))
title = 'Regress Gaussian'
create_and_plot_points_in_data_set(training_data_set, training_data_set_filename, (1, -1), title)
plot_madge_data(data, filename, title)

