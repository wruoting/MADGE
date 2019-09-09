from __future__ import print_function

import plygdata as pg

# Or, you can 'import' classes and functions directly like this:
# from plygdata.datahelper import DatasetType
# from plygdata.dataset import generate

data_noise=0.0
validation_data_ratio = 0

data_array = pg.generate_data(pg.DatasetType.ClassifyCircleData, data_noise)

# Store data array in file
f= open("./SampleData/circle_classification.txt","w+")
f.write(str(data_array))
## If we want to see it normally
# Divide the data for training and validating at a specified ratio (further, separate each data into Coordinate point data part and teacher label part)
X_train, y_train, X_valid, y_valid = pg.split_data(data_array, validation_size=validation_data_ratio)

# Plot the data on the standard graph for Playground
fig, ax = pg.plot_points_with_playground_style(X_train, y_train, X_valid, y_valid, figsize = (6, 6), dpi = 100)

import matplotlib.pyplot as plt
plt.show()