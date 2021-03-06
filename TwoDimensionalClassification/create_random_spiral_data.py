from __future__ import print_function

import plygdata as pg
import numpy as np


# Store data array in file
def write_to_file(filename, data):
    f = open("../SampleData/{}.txt".format(filename), "w+")
    f.write(str(data))
    f.close()

# Or, you can 'import' classes and functions directly like this:
# from plygdata.datahelper import DatasetType
# from plygdata.dataset import generate


def make_array_file(x, y):
    data_noise = 0.0
    validation_data_ratio = 0

    ClassifySpiralData = pg.generate_data(pg.DatasetType.ClassifySpiralData, data_noise)

    # Multiply each 0 coordinate by a large number, in this case 426,561
    # Multiply each 1 coordinate by a smaller number, in this case 15, 624
    new_classify_spiral_data = []

    for element in ClassifySpiralData:
        new_classify_spiral_data.append(list(np.multiply(element, [x, y, 1])))

    data_array = new_classify_spiral_data
    data_array_str = 'ClassifySpiralDataNonSquare-{}-{}'.format(x, y)
    write_to_file(data_array_str, data_array)

