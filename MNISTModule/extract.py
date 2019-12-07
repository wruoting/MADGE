from MNISTModule.classification_set import ClassificationSet
from MNISTModule.classification import Classification
from mnist.loader import MNIST
import numpy as np
import os

samples_path = './SampleData/MNIST'
write_path = './accuracy.txt'
mndata = MNIST(samples_path)


mnist_training_set = ClassificationSet()

if os.path.exists(write_path):
    mode = 'a+'
else:
    f= open(write_path,"w+")
for test_sigma in np.arange(0.82, 0.87, 0.001):
    images_training, labels_training = mndata.load_training()
    images_testing, labels_testing = mndata.load_testing()
    classification = Classification(images_training, labels_training, images_testing, labels_testing, sigma=test_sigma)
    classification.load_model(path='./MNISTModule/')
    with open(write_path, 'a+') as file:
        file.write(str(test_sigma) + ',' + classification.calculate_accuracy(calculate=False, mode='return'))
        file.write('\n')
