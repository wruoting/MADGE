import plygdata as pg
import numpy as np
import ast


def create_iris_data(validation_data_ratio):
    iris_path = './SampleData/IrisData/iris.data'
    iris_data = []
    iris_classifications = []
    with open(iris_path) as file:
        for entry in file.readlines():
            entry_array = entry.strip('\n').split(',')
            iris_classifications.append(entry_array[-1])
            iris_data.append(entry_array)
    unique_classifiers = np.unique(iris_classifications)
    classification_mapping = {}
    for index, classifier in enumerate(unique_classifiers):
        classification_mapping[classifier] = index + 1
    X_train, Y_train, X_validate, Y_validate = pg.split_data(iris_data, validation_size=validation_data_ratio)
    X_train = X_train.astype(np.float)
    X_validate = X_validate.astype(np.float)
    Y_train_mapped = list(map(lambda classification: classification_mapping[classification[0]], Y_train))
    Y_validate_mapped = list(map(lambda classification: classification_mapping[classification[0]], Y_validate))

    return X_train, Y_train_mapped, X_validate, Y_validate_mapped, classification_mapping