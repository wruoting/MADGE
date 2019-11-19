import plygdata as pg
import numpy as np
import ast


def create_abalone_data(validation_data_ratio):
    abalone_path = './SampleData/AbaloneData/abalone.data'
    abalone_data = []
    abalone_classifications = []
    # Abalone data sex needs to be mapped
    mapping = {
        'M': 1,
        'F': 2,
        'I': 3,
    }
    with open(abalone_path) as file:
        for entry in file.readlines():
            entry_array = entry.strip('\n').split(',')
            abalone_classifications.append(entry_array[-1])
            mapped_array = []
            for index, element in enumerate(entry_array):
                if index == 0:
                    mapped_array.append(mapping[element])
                else:
                    mapped_array.append(element)
            abalone_data.append(mapped_array)

    X_train, Y_train, X_validate, Y_validate = pg.split_data(abalone_data, validation_size=validation_data_ratio)
    X_train = X_train.astype(np.float)
    X_validate = X_validate.astype(np.float)
    Y_train = Y_train.astype(np.float)
    Y_validate = Y_validate.astype(np.float)

    return X_train, Y_train, X_validate, Y_validate