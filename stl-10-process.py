import numpy as np
from MNISTModule.madge_calculator import classify_by_distance

# image shape
from MNISTModule.classification import Classification

HEIGHT = 96
WIDTH = 96
DEPTH = 3

# size of a single image in bytes
SIZE = HEIGHT * WIDTH * DEPTH

# path to the binary train file with image data
DATA_PATH_TRAIN = './data/stl10_binary/train_X.bin'
LABEL_PATH_TRAIN = './data/stl10_binary/train_y.bin'
DATA_PATH_TEST = './data/stl10_binary/test_X.bin'
LABEL_PATH_TEST = './data/stl10_binary/test_y.bin'


# test to check if the image is read correctly
def load_data_sets(data_path, label_path):
    with open(data_path) as f:
        # Open up every image in a vector
        everything = np.fromfile(f, dtype=np.uint8)
        images = np.reshape(everything, (-1, 3, 1, 96 * 96))
        # For each image, create a set of layers, one for RGB
        red_training, green_training, blue_training = [], [], []
        for image in images:
            red_training.append(image[0])
            green_training.append(image[1])
            blue_training.append(image[2])

    with open(label_path) as f:
        label_vector = np.fromfile(f, dtype=np.uint8)
    return red_training, green_training, blue_training, label_vector


# Instead of 96 x 96, we're going to crossfold with 5 x 5 swaths and create a vector like this
def load_data_sets_cross_fold(data_path, label_path, cross_fold=5):
    cross_fold_vector = cross_fold * cross_fold
    with open(data_path) as f:
        # Open up every image in a vector
        everything = np.fromfile(f, dtype=np.uint8)
        images = np.reshape(everything, (-1, 3, 1, 96 * 96))

    with open(label_path) as f:
        label_vector = np.fromfile(f, dtype=np.uint8)


    for image, label in zip(images, label_vector):
        # Break down each image into cross_fold x cross_fold swaths
        red_image = image[0]
        green_image = image[1]
        blue_image = image[2]
        for index in red_image[0:-cross_fold_vector]:
            [index:index+cross_fold_vector]

red_training, green_training, blue_training, label_train = load_data_sets(DATA_PATH_TRAIN, LABEL_PATH_TRAIN)

red_testing, green_testing, blue_testing, label_test = load_data_sets(DATA_PATH_TEST, LABEL_PATH_TEST)

# with open('All_Accuracy.txt', "w+") as f:
#     for value_all in np.arange(1.5, 5, 0.3):
#         f.write(str(value_all) + '\n')
#         f.write(Classification(all_training, label_train, all_testing, label_test, sigma=value_all)
#                 .calculate_accuracy(mode='verbose'))
# Classification(all_training, label_train, all_testing, label_test, sigma=3).calculate_accuracy(mode='verbose')
blue_classification = Classification(blue_training, label_train, blue_testing, label_test, sigma=2.8)\
    .calculate_accuracy(mode='verbose')

red_classification = Classification(red_training, label_train, red_testing, label_test, sigma=2.8)\
    .calculate_accuracy(mode='verbose')


green_classification = Classification(green_training, label_train, green_testing, label_test, sigma=2.8)\
    .calculate_accuracy(mode='verbose')


#
# all_classifications = np.divide(
#     np.sum(np.array(red_classification, green_classification, blue_classification), axis=0), 3)
# labels = np.unique(label_train)
# final_classifications = []
# for fc in all_classifications:
#     final_classifications.append(classify_by_distance(labels, fc))
#
# for test_classification, resulting_classification in zip(label_test, final_classifications):
#     print('Real')
#     print(test_classification)
#     print('Test')
#     print(resulting_classification)