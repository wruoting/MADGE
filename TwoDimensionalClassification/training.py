from TwoDimensionalClassification.graph import read_data_from_file, convert_array_to_array_of_tuples
from TwoDimensionalClassification.point import Point
from TwoDimensionalClassification.classification_set import ClassificationSet
from TwoDimensionalClassification.classification_set_n import ClassificationSetN
from TwoDimensionalClassification.madge_calculator import classify_by_distance
import plygdata as pg
import numpy as np
import plotly as py
import plotly.graph_objects as go
from scipy import interpolate
import plotly.figure_factory as ff


def classify_data(path, classifiers, linspace, validation_data_ratio=0.2, generate_graphs=True,
                  surface_name='MADGE Surface', title='Classification Data', filename='Test.html'):
    # Make the lower classifier always the first
    if classifiers[0] > classifiers[1]:
        classifiers = (classifiers[1], classifiers[0])
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
    Z_validate_interpolate = f_training_interpolate(x_test, y_test)
    
    # If the z value is above 0, it is classified as the greater of the two classifications, 
    # if it is below 0, it is classified as the less of the two classifications
    # These classifications are arbitrary
    # We compare these to Y_validate for an accuracy
    def compare_with_zero(value):
        if value > 0:
            return classifiers[1]
        else:
            return classifiers[0]
    test_classification_results = list(map(compare_with_zero, Z_validate_interpolate))
    
    correct_results = 0
    for result, test_result in zip(test_classification_results, Y_validate):
        if result == test_result:
            correct_results += 1
    accuracy = np.divide(correct_results, len(test_classification_results))
    if generate_graphs:
        # Now we generate graphs 
        trace_surface = go.Surface(x=X, y=Y, z=Z, name=surface_name, showscale=False)
        trace_scatter_class_a_training = go.Scatter3d(x=x_0_train, y=y_0_train, z=z_0_train, mode='markers',
                                             name='Training Classifier {}'.format(classifiers[0]), 
                                             marker=dict(
                                                 size=3,
                                                 color='#f29938',    
                                                 opacity=1
                                             ), legendgroup="Group_Train")
        trace_scatter_class_b_training = go.Scatter3d(x=x_1_train, y=y_1_train, z=z_1_train, mode='markers',
                                             name='Training Classifier {}'.format(classifiers[1]),
                                             marker=dict(
                                                 size=3,
                                                 color='#257ec0',    
                                                 opacity=1
                                             ), legendgroup="Group_Train")
        # Add plots for testing data
        trace_scatter_class_a_testing= go.Scatter3d(x=x_0_test, y=y_0_test, z=z_0_test, mode='markers',
                                             name='Testing Classifier {}'.format(classifiers[0]),
                                             marker=dict(
                                                 size=3,
                                                 color='#f29938',    
                                                 opacity=0.4
                                             ), legendgroup="Group_Test")
                                        
        trace_scatter_class_b_testing = go.Scatter3d(x=x_1_test, y=y_1_test, z=z_1_test, mode='markers',
                                             name='Testing Classifier {}'.format(classifiers[1]),
                                             marker=dict(
                                                 size=3,
                                                 color='#257ec0',    
                                                 opacity=0.4
                                             ), legendgroup="Group_Test")
        
        # Append the title name with accuracy
        title = title + "\nAccuracy: {}".format(accuracy)
        data = [trace_surface, 
                trace_scatter_class_a_training, trace_scatter_class_b_training, 
                trace_scatter_class_a_testing, trace_scatter_class_b_testing]
        fig = go.Figure(data=data)
        fig.update_layout(title=title, autosize=True,
                          width=700, height=700,
                          margin=dict(l=50, r=50, b=65, t=90))
        py.offline.plot(fig, filename=filename)
    else:
        # If we're not generating graphs we will return the accuracy
        return accuracy


def classify_data_by_point(path, classifiers, linspace, validation_data_ratio=0.2, generate_graphs=True,
                           surface_name='MADGE Surface', title='Classification Data', filename='Test.html',
                           normalization_standard_deviation_factor=6):
    # Make the lower classifier always the first
    if classifiers[0] > classifiers[1]:
        classifiers = (classifiers[1], classifiers[0])
    # Create our training data params
    training_data_set = convert_array_to_array_of_tuples(read_data_from_file(path))
    ## If we want to see it normally
    # Divide the data for training and validating at a specified ratio (further, separate each data into Coordinate point data part and teacher label part)
    X_train, Y_train, X_validate, Y_validate = pg.split_data(training_data_set, validation_size=validation_data_ratio)

    # # This creates the plane with the data we are working with
    new_set = ClassificationSetN()
    x_0_train, y_0_train, z_0_train, x_1_train, y_1_train, z_1_train = [], [], [], [], [], []
    train_label_dim = 2
    train_label_sigma_max = np.zeros(train_label_dim)
    train_label_sigma_min = np.zeros(train_label_dim)
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
        if train_x > train_label_sigma_max[0]:
            train_label_sigma_max[0] = train_x
        if train_y < train_label_sigma_min[1]:
            train_label_sigma_min[1] = train_y
    new_set.range_vector = np.subtract(train_label_sigma_max, train_label_sigma_min)  # range(w)
    new_set.normalization_standard_deviation_factor = normalization_standard_deviation_factor
    new_set.range_vector = np.divide(new_set.range_vector, new_set.normalization_standard_deviation_factor)
    if generate_graphs:
        # https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
        x_space = np.linspace(linspace[0], linspace[1], linspace[2])
        y_space = np.linspace(linspace[0], linspace[1], linspace[2])
        X, Y = np.meshgrid(x_space, y_space)
        Z = []
        for x_array, y_array in zip(X, Y):
            z_point = []
            for x_point, y_point in zip(x_array, y_array):
                predicted_point = np.round(new_set.calculate_madge_data_and_map_to_point(Point(x_point, y_point), normalize=True))
                if np.absolute(classifiers[0] - predicted_point) > np.absolute(classifiers[1] - predicted_point):
                    z_point.append(classifiers[1])
                else:
                    z_point.append(classifiers[0])
            Z.append(z_point)
        Z = np.array(Z)

    # This creates the testing data graph data
    x_0_test, y_0_test, z_0_test, x_1_test, y_1_test, z_1_test, x_test, y_test, z_test = \
        [], [], [], [], [], [], [], [], []
    for [test_x, test_y], classification in zip(X_validate, Y_validate):
        x_test.append(test_x)
        y_test.append(test_y)
        z_test.append(np.round(new_set.calculate_madge_data_and_map_to_point(Point(test_x, test_y), normalize=True)))
        if classification == classifiers[0]:
            x_0_test.append(test_x)
            y_0_test.append(test_y)
            z_0_test.append(0)
        elif classification == classifiers[1]:
            x_1_test.append(test_x)
            y_1_test.append(test_y)
            z_1_test.append(0)

    correct_results = 0
    for result, test_result in zip(z_test, Y_validate):
        if result == test_result:
            correct_results += 1
    accuracy = np.divide(correct_results, len(z_test))
    if generate_graphs:
        # Now we generate graphs
        trace_surface = go.Surface(x=X, y=Y, z=Z, name=surface_name, showscale=False)
        trace_scatter_class_a_training = go.Scatter3d(x=x_0_train, y=y_0_train, z=z_0_train, mode='markers',
                                                      name='Training Classifier {}'.format(classifiers[0]),
                                                      marker=dict(
                                                          size=3,
                                                          color='#f29938',
                                                          opacity=1
                                                      ), legendgroup="Group_Train")
        trace_scatter_class_b_training = go.Scatter3d(x=x_1_train, y=y_1_train, z=z_1_train, mode='markers',
                                                      name='Training Classifier {}'.format(classifiers[1]),
                                                      marker=dict(
                                                          size=3,
                                                          color='#257ec0',
                                                          opacity=1
                                                      ), legendgroup="Group_Train")
        # Add plots for testing data
        trace_scatter_class_a_testing = go.Scatter3d(x=x_0_test, y=y_0_test, z=z_0_test, mode='markers',
                                                     name='Testing Classifier {}'.format(classifiers[0]),
                                                     marker=dict(
                                                         size=3,
                                                         color='#f29938',
                                                         opacity=0.4
                                                     ), legendgroup="Group_Test")

        trace_scatter_class_b_testing = go.Scatter3d(x=x_1_test, y=y_1_test, z=z_1_test, mode='markers',
                                                     name='Testing Classifier {}'.format(classifiers[1]),
                                                     marker=dict(
                                                         size=3,
                                                         color='#257ec0',
                                                         opacity=0.4
                                                     ), legendgroup="Group_Test")

        # Append the title name with accuracy
        title = title + "\nAccuracy: {}".format(accuracy)
        data = [trace_surface,
                trace_scatter_class_a_training, trace_scatter_class_b_training,
                trace_scatter_class_a_testing, trace_scatter_class_b_testing]
        fig = go.Figure(data=data)
        fig.update_layout(title=title, autosize=True,
                          width=700, height=700,
                          margin=dict(l=50, r=50, b=65, t=90))
        py.offline.plot(fig, filename=filename)
    else:
        # If we're not generating graphs we will return the accuracy
        return accuracy


# This function takes set split data and runs point analysis with no graphing
def classify_data_by_point_set_validate(path, classifiers, split_data, normalization_standard_deviation_factor=6):
    # Make the lower classifier always the first
    if classifiers[0] > classifiers[1]:
        classifiers = (classifiers[1], classifiers[0])
    # Create our training data params
    training_data_set = convert_array_to_array_of_tuples(read_data_from_file(path))
    ## If we want to see it normally
    # Divide the data for training and validating at a specified ratio (further, separate each data into Coordinate point data part and teacher label part)
    X_train, Y_train, X_validate, Y_validate = split_data[0], split_data[1], split_data[2], split_data[3]

    # # This creates the plane with the data we are working with
    new_set = ClassificationSetN()
    x_0_train, y_0_train, z_0_train, x_1_train, y_1_train, z_1_train = [], [], [], [], [], []
    train_label_dim = 2
    train_label_sigma_max = np.zeros(train_label_dim)
    train_label_sigma_min = np.zeros(train_label_dim)
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
        if train_x > train_label_sigma_max[0]:
            train_label_sigma_max[0] = train_x
        if train_y < train_label_sigma_min[1]:
            train_label_sigma_min[1] = train_y
    new_set.range_vector = np.subtract(train_label_sigma_max, train_label_sigma_min)  # range(w)
    new_set.normalization_standard_deviation_factor = normalization_standard_deviation_factor

    # This creates the testing data graph data
    x_0_test, y_0_test, z_0_test, x_1_test, y_1_test, z_1_test, x_test, y_test, z_test = \
        [], [], [], [], [], [], [], [], []
    for [test_x, test_y], classification in zip(X_validate, Y_validate):
        x_test.append(test_x)
        y_test.append(test_y)
        z_test.append(np.round(new_set.calculate_madge_data_and_map_to_point(Point(test_x, test_y), normalize=True)))
        if classification == classifiers[0]:
            x_0_test.append(test_x)
            y_0_test.append(test_y)
            z_0_test.append(0)
        elif classification == classifiers[1]:
            x_1_test.append(test_x)
            y_1_test.append(test_y)
            z_1_test.append(0)

    correct_results = 0
    for result, test_result in zip(z_test, Y_validate):
        if result == test_result:
            correct_results += 1
    accuracy = np.divide(correct_results, len(z_test))
    return accuracy


# This function takes set split data and runs point analysis with no graphing
def classify_data_by_point_set_validate_v2(path, classifiers, split_data, sigma=6):
    # Make the lower classifier always the first
    if classifiers[0] > classifiers[1]:
        classifiers = (classifiers[1], classifiers[0])
    # Create our training data params
    training_data_set = convert_array_to_array_of_tuples(read_data_from_file(path))
    ## If we want to see it normally
    # Divide the data for training and validating at a specified ratio (further, separate each data into Coordinate point data part and teacher label part)
    X_train, Y_train, X_validate, Y_validate = split_data[0], split_data[1], split_data[2], split_data[3]

    # # This creates the plane with the data we are working with
    new_set = ClassificationSetN()
    x_0_train, y_0_train, z_0_train, x_1_train, y_1_train, z_1_train = [], [], [], [], [], []
    train_label_dim = 2
    train_label_sigma_max = np.zeros(train_label_dim)
    train_label_sigma_min = np.zeros(train_label_dim)
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
        if train_x > train_label_sigma_max[0]:
            train_label_sigma_max[0] = train_x
        if train_y < train_label_sigma_min[1]:
            train_label_sigma_min[1] = train_y
    new_set.range_vector = np.subtract(train_label_sigma_max, train_label_sigma_min)  # range(w)
    # print('Sigma range vector: {}'.format(np.divide(new_set.range_vector, sigma)))
    print('Range vector: {}'.format(new_set.range_vector))
    # This creates the testing data graph data
    x_0_test, y_0_test, z_0_test, x_1_test, y_1_test, z_1_test, x_test, y_test, z_test = \
        [], [], [], [], [], [], [], [], []
    for [test_x, test_y], classification in zip(X_validate, Y_validate):
        x_test.append(test_x)
        y_test.append(test_y)
        classification_result = classify_by_distance(
                classifiers,
                new_set.calculate_madge_data_and_map_to_point_v2(Point(test_x, test_y), sigma=sigma))
        z_test.append(classification_result)
        if classification == classifiers[0]:
            x_0_test.append(test_x)
            y_0_test.append(test_y)
            z_0_test.append(0)
        elif classification == classifiers[1]:
            x_1_test.append(test_x)
            y_1_test.append(test_y)
            z_1_test.append(0)

    correct_results = 0
    for result, test_result in zip(z_test, Y_validate):
        if result == test_result:
            correct_results += 1
    accuracy = np.divide(correct_results, len(z_test))
    return accuracy


def classify_data_by_point_set_validate_v3(path, classifiers, split_data, sigma, path_to_write):
    # Make the lower classifier always the first
    if classifiers[0] > classifiers[1]:
        classifiers = (classifiers[1], classifiers[0])

    # Divide the data for training and validating at a specified ratio (further, separate each data into Coordinate point data part and teacher label part)
    X_train, Y_train, X_validate, Y_validate = split_data[0], split_data[1], split_data[2], split_data[3]

    # # This creates the plane with the data we are working with
    new_set = ClassificationSetN()
    x_0_train, y_0_train, z_0_train, x_1_train, y_1_train, z_1_train = [], [], [], [], [], []
    train_label_dim = 2
    train_label_sigma_max = np.zeros(train_label_dim)
    train_label_sigma_min = np.zeros(train_label_dim)
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
        if train_x > train_label_sigma_max[0]:
            train_label_sigma_max[0] = train_x
        if train_y < train_label_sigma_min[1]:
            train_label_sigma_min[1] = train_y
    new_set.range_vector = np.subtract(train_label_sigma_max, train_label_sigma_min)  # range(w)
    # This creates the testing data graph data
    x_0_test, y_0_test, z_0_test, x_1_test, y_1_test, z_1_test, x_test, y_test, z_test = \
        [], [], [], [], [], [], [], [], []
    for [test_x, test_y], classification in zip(X_validate, Y_validate):
        x_test.append(test_x)
        y_test.append(test_y)
        classification_result = classify_by_distance(
                classifiers,
                new_set.calculate_madge_data_and_map_to_point_v3(Point(test_x, test_y), sigma=sigma))
        z_test.append(classification_result)
        if classification == classifiers[0]:
            x_0_test.append(test_x)
            y_0_test.append(test_y)
            z_0_test.append(0)
        elif classification == classifiers[1]:
            x_1_test.append(test_x)
            y_1_test.append(test_y)
            z_1_test.append(0)

    correct_results = 0
    for result, test_result in zip(z_test, Y_validate):
        if result == test_result:
            correct_results += 1
    accuracy = np.divide(correct_results, len(z_test))
    with open(path_to_write, 'a+') as file:
        file.write('{},{},{}\n'.format(sigma[0], sigma[1], accuracy))
    return accuracy


# This function uses the gaussian weights
def classify_data_by_point_set_validate_v4(classifiers, split_data, sigma, path_to_write):
    # Make the lower classifier always the first
    if classifiers[0] > classifiers[1]:
        classifiers = (classifiers[1], classifiers[0])
    # Divide the data for training and validating at a specified ratio (further, separate each data into Coordinate point data part and teacher label part)
    X_train, Y_train, X_validate, Y_validate = split_data[0], split_data[1], split_data[2], split_data[3]

    # # This creates the plane with the data we are working with
    new_set = ClassificationSetN()
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

    new_set.range_vector = [np.subtract(np.amax(X_train), np.amin(X_train)),
                            np.subtract(np.amax(Y_train), np.amin(Y_train))]
    # range(w)
    # This creates the testing data graph data
    x_0_test, y_0_test, z_0_test, x_1_test, y_1_test, z_1_test, x_test, y_test, z_test = \
        [], [], [], [], [], [], [], [], []
    for [test_x, test_y], classification in zip(X_validate, Y_validate):
        x_test.append(test_x)
        y_test.append(test_y)
        classification_result = classify_by_distance(
                classifiers,
                new_set.calculate_madge_data_and_map_to_point_v4(Point(test_x, test_y), sigma=sigma))
        print('classification_result', classification_result)
        print('classification', classification)
        print('--------------')
        z_test.append(classification_result)
        if classification == classifiers[0]:
            x_0_test.append(test_x)
            y_0_test.append(test_y)
            z_0_test.append(0)
        elif classification == classifiers[1]:
            x_1_test.append(test_x)
            y_1_test.append(test_y)
            z_1_test.append(0)

    correct_results = 0
    for result, test_result in zip(z_test, Y_validate):
        if result == test_result:
            correct_results += 1
    accuracy = np.divide(correct_results, len(z_test))
    # with open(path_to_write, 'a+') as file:
    #     file.write('{},{},{}\n'.format(sigma[0], sigma[1], accuracy))
    return accuracy


# What if we normalize dimensions to be square
def classify_data_by_point_set_validate_v5(classifiers, split_data, sigma):
    # Make the lower classifier always the first
    if classifiers[0] > classifiers[1]:
        classifiers = (classifiers[1], classifiers[0])
    # Divide the data for training and validating at a specified ratio
    # (further, separate each data into Coordinate point data part and teacher label part)
    X_train, Y_train, X_validate, Y_validate = split_data[0], split_data[1], split_data[2], split_data[3]

    # Figure out what the range is, and then shrink to normalize them
    range_vector = np.subtract(np.amax(X_train, 0), np.amin(X_train, 0))
    X_train_normalize = np.divide(X_train, range_vector)
    X_validate_normalize = np.divide(X_validate, range_vector)

    # # This creates the plane with the data we are working with
    new_set = ClassificationSetN()
    x_0_train, y_0_train, z_0_train, x_1_train, y_1_train, z_1_train = [], [], [], [], [], []

    for [train_x, train_y], classification in zip(X_train_normalize, Y_train):
        new_set.add_point(Point(train_x, train_y, classification))
        if classification == classifiers[0]:
            x_0_train.append(train_x)
            y_0_train.append(train_y)
            z_0_train.append(0)
        elif classification == classifiers[1]:
            x_1_train.append(train_x)
            y_1_train.append(train_y)
            z_1_train.append(0)

    # This creates the testing data graph data
    x_0_test, y_0_test, z_0_test, x_1_test, y_1_test, z_1_test, x_test, y_test, z_test = \
        [], [], [], [], [], [], [], [], []
    for [test_x, test_y], classification in zip(X_validate_normalize, Y_validate):
        x_test.append(test_x)
        y_test.append(test_y)
        classification_result = classify_by_distance(
                classifiers,
                new_set.calculate_madge_data_and_map_to_point_v5(Point(test_x, test_y), sigma))
        z_test.append(classification_result)
        if classification == classifiers[0]:
            x_0_test.append(test_x)
            y_0_test.append(test_y)
            z_0_test.append(0)
        elif classification == classifiers[1]:
            x_1_test.append(test_x)
            y_1_test.append(test_y)
            z_1_test.append(0)

    correct_results = 0
    for result, test_result in zip(z_test, Y_validate):
        if result == test_result:
            correct_results += 1
    accuracy = np.divide(correct_results, len(z_test))
    return accuracy


def test_scatter(X, Y, Z, x_test, y_test, Z_validate, surface_name='MADGE Interpolate Surface', filename='test.html'):
    """
    Generates a plot of the interpolated data overlayed with the general data
    :param X: 
    :return: None
    """
    trace_surface = go.Surface(x=X, y=Y, z=Z, name=surface_name)
    test_scatter = go.Scatter3d(x=x_test, y=y_test, z=Z_validate, mode='markers',
                                         name='Interpolated Values')
    data = [trace_surface, test_scatter]
    fig = go.Figure(data=data)
    py.offline.plot(fig, filename=filename)


def graph_bar(data, filename):
    fig = ff.create_distplot(data, ['Accuracy Frequency'])
    py.offline.plot(fig, filename=filename)
