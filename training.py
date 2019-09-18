from graph import read_data_from_file, convert_array_to_array_of_tuples
from point import Point
from classification_set import ClassificationSet
import matplotlib.pyplot as plt
import plygdata as pg
import numpy as np
import plotly as py
import plotly.graph_objects as go
from scipy import interpolate

def classify_data(path, classifiers, linspace, validation_data_ratio=0.2, surface_name='MADGE Surface'):

    # Create our training data params
    training_data_set = convert_array_to_array_of_tuples(read_data_from_file(path))
    ## If we want to see it normally
    # Divide the data for training and validating at a specified ratio (further, separate each data into Coordinate point data part and teacher label part)
    X_train, Y_train, X_validate, Y_validate = pg.split_data(training_data_set, validation_size=validation_data_ratio)

    # # This creates the plane with the data we are working with
    new_set = ClassificationSet(sigma=1)
    x_0_train, y_0_train, z_0_train, x_1_train, y_1_train, z_1_train = [], [], [], [], [], []
    for [train_x, train_y], classification in zip(X_train, Y_train):
        new_set.add_point(Point(train_x, train_y, classification))
        if classification == classifiers[0]:
            x_0_train.append(train_x)
            y_0_train.append(train_y)
            z_0_train.append(0)
        elif classification == classifiers[1]:
            x_1_train.append(train_x)
            y_1_train.append(train_y)
            z_1_train.append(0)
    x_space = np.linspace(linspace[0], linspace[1], linspace[2])
    y_space = np.linspace(linspace[0], linspace[1], linspace[2])
    X, Y = np.meshgrid(x_space, y_space)
    Z = new_set.calculate_madge_data_and_map_to_plane(X, Y)
    
    ## This creates the testing data graph data
    x_0_test, y_0_test, z_0_test, x_1_test, y_1_test, z_1_test, x_test, y_test = [], [], [], [], [], [], [], []
    for [test_x, test_y], classification in zip(X_validate, Y_validate):
        x_test.append(test_x)
        y_test.append(test_y)
        if classification == classifiers[0]:
            x_0_test.append(test_x)
            y_0_test.append(test_y)
            z_0_test.append(0)
        elif classification == classifiers[1]:
            x_1_test.append(test_x)
            y_1_test.append(test_y)
            z_1_test.append(0)
    # Output an accuracy
    # This will be done via interpolation of the graph
    # We will create an interp function given an x,y array, and output the interpolated vector
    # Input vector will be [X_validation, Y_validation]
    # Our interp function will be the new_set.calculate_madge_data_and_map_to_plane function
    # TODO: is cubic spline 2d interpolation the best to use?
    # Ummm? https://stackoverflow.com/questions/37872171/how-can-i-perform-two-dimensional-interpolation-using-scipy
    # I have no idea what the fuck RBF is but let's just say for now that it works and dear god that's amazing
    f_training_interpolate = interpolate.Rbf(X, Y, Z, function='cubic', smooth=0)
    Z_validate = f_training_interpolate(x_test, y_test)
    trace_surface = go.Surface(x=X, y=Y, z=Z, name=surface_name)
    test_scatter = go.Scatter3d(x=x_test, y=y_test, z=Z_validate, mode='markers',
                                         name='Interpolated Values'.format(classifiers[0]))
    data = [trace_surface, test_scatter]
    fig = go.Figure(data=data)
    py.offline.plot(fig, filename='test.html')
    
    # Now we generate graphs 
    # trace_surface = go.Surface(x=X, y=Y, z=Z, name=surface_name)
    # trace_scatter_class_a_training = go.Scatter3d(x=x_0_train, y=y_0_train, z=z_0_train, mode='markers',
    #                                      name='Training Classifier {}'.format(classifiers[0]))
    # trace_scatter_class_b_training = go.Scatter3d(x=x_1_train, y=y_1_train, z=z_1_train, mode='markers',
    #                                      name='Training Classifier {}'.format(classifiers[1]))
    # # Add plots for testing data
    # trace_scatter_class_a_testing= go.Scatter3d(x=x_0_test, y=y_0_test, z=z_0_test, mode='markers',
    #                                      name='Testing Classifier {}'.format(classifiers[0]))
    # trace_scatter_class_b_testing = go.Scatter3d(x=x_1_test, y=y_1_test, z=z_1_test, mode='markers',
    #                                      name='Testing Classifier {}'.format(classifiers[1]))
    # 
    # data = [trace_surface, 
    #         trace_scatter_class_a_training, trace_scatter_class_b_training, 
    #         trace_scatter_class_a_testing, trace_scatter_class_b_testing]



linspace = (-6, 6, 30)
classify_data("./SampleData/ClassifySpiralData.txt", (-1, 1), linspace, validation_data_ratio=0.2)