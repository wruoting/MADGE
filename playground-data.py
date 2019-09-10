from __future__ import print_function

import plygdata as pg

# Store data array in file
def write_to_file(filename, data):
    f= open("./SampleData/{}.txt".format(filename),"w+")
    f.write(str(data))
    f.close()

# Or, you can 'import' classes and functions directly like this:
# from plygdata.datahelper import DatasetType
# from plygdata.dataset import generate

data_noise=0.0
validation_data_ratio = 0

ClassifyCircleData = pg.generate_data(pg.DatasetType.ClassifyCircleData, data_noise)
ClassifyXORData = pg.generate_data(pg.DatasetType.ClassifyXORData, data_noise)
ClassifyTwoGaussData = pg.generate_data(pg.DatasetType.ClassifyTwoGaussData, data_noise)
ClassifySpiralData = pg.generate_data(pg.DatasetType.ClassifySpiralData, data_noise)
RegressPlane = pg.generate_data(pg.DatasetType.RegressPlane, data_noise)
RegressGaussian = pg.generate_data(pg.DatasetType.RegressGaussian, data_noise)

data_array = [ClassifyCircleData, ClassifyXORData, ClassifyTwoGaussData, 
                ClassifySpiralData, RegressPlane, RegressGaussian]
data_array_str = ['ClassifyCircleData', 'ClassifyXORData', 'ClassifyTwoGaussData', 
                'ClassifySpiralData', 'RegressPlane', 'RegressGaussian']

for element, element_str in zip(data_array, data_array_str):
    write_to_file(element_str, element)

# ## If we want to see it normally
# # Divide the data for training and validating at a specified ratio (further, separate each data into Coordinate point data part and teacher label part)
# X_train, y_train, X_valid, y_valid = pg.split_data(data_array, validation_size=validation_data_ratio)
# 
# # Plot the data on the standard graph for Playground
# fig, ax = pg.plot_points_with_playground_style(X_train, y_train, X_valid, y_valid, figsize = (6, 6), dpi = 100)
# 
# import matplotlib.pyplot as plt
# plt.show()