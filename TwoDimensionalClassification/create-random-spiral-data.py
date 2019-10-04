from __future__ import print_function

import plygdata as pg


# Store data array in file
def write_to_file(filename, data):
    f = open("./SampleData/{}.txt".format(filename),"w+")
    f.write(str(data))
    f.close()

# Or, you can 'import' classes and functions directly like this:
# from plygdata.datahelper import DatasetType
# from plygdata.dataset import generate


data_noise=0.0
validation_data_ratio = 0

ClassifySpiralData = pg.generate_data(pg.DatasetType.ClassifySpiralData, data_noise)

data_array = ClassifySpiralData
data_array_str = 'ClassifySpiralData'
#
# for element, element_str in zip(data_array, data_array_str):
#     write_to_file(element_str, element)
