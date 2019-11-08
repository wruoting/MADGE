from MNISTModule.classification_set import ClassificationSet
from MNISTModule.madge_calculator import classify_by_distance
from MNISTModule.n_point import NPoint

import numpy as np
from mnist.loader import MNIST

samples_path = './SampleData/MNIST'

mndata = MNIST(samples_path)
images_training, labels_training = mndata.load_training()
images_testing, labels_testing = mndata.load_testing()

mnist_training_set = ClassificationSet()

# 1. Each point needs to look for a max and a min for sigmas. Runtime unfortunately N*M
print('Processing images')
# Figure out what the range is, and then shrink to normalize them


def replace_with_one(a):
    if a == 0:
        return 1
    return a


replace_with_one_vector = np.vectorize(replace_with_one)
range_vector = replace_with_one_vector(np.subtract(np.amax(images_training, 0), np.amin(images_training, 0)))

images_training_normalize = np.divide(images_training, range_vector)
images_testing_normalize = np.divide(images_testing, range_vector)

# We are going to calculate a sigma here that is proportional to the average of the range of the points
# This calculation will allow us to reflect gaussian area with a relative rather than absolute distance using
# any arbitrary sigma.
# 2. Take the length of the training/testing labels. We can shortcut and take the first one
#    We can then make an empty array of zeros with that as the distance

for image, label in zip(images_training_normalize, labels_training):
    mnist_training_set.add_point(NPoint(image, type=label))
# 3. Calculate the average sigma and use this ubiquitously
# We are going to use the equation sum(n_i/sum(n) * range(w_i)/6),
# where n_i is the ith dimension, sum(n) is the sum of the range of all dimensions
# w is the range of the dimension at i

match = 0
total = len(images_testing)
index_testing = 0
sigma = 0.05 * len(labels_training)
print('Running Testing Data')
for image, label in zip(images_testing_normalize, labels_testing):
    distance = mnist_training_set.calculate_madge_data_and_map_to_point(NPoint(image, type=label), sigma)
    if label == classify_by_distance(
            np.unique(labels_training),
            distance):
        match = match + 1
    index_testing = index_testing + 1
    print('Processing {} out of {}'.format(index_testing, 100))
    print("Test")
    print(label)
    print('Real')
    print(mnist_training_set.calculate_madge_data_and_map_to_point(NPoint(image, type=label), sigma))
    if index_testing == 100:
        break

with open('Accuracy.txt', "w+") as f:
    f.write(str(np.divide(match, index_testing)))
