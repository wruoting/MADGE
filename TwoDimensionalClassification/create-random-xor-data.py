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


data_noise=0.0
validation_data_ratio = 0

ClassifyXORData = pg.generate_data(pg.DatasetType.ClassifyXORData, data_noise)

# Multiply each 0 coordinate by a large number, in this case 426,561
# Multiply each 1 coordinate by a smaller number, in this case 15, 624
new_classify_xor_data = []
for element in ClassifyXORData:
    new_classify_xor_data.append(list(np.multiply(element, [300, 7, 1])))

data_array = new_classify_xor_data
data_array_str = 'ClassifyXORDataNonSquare'
write_to_file(data_array_str, data_array)

