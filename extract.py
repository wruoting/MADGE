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
test_sigma = 0.85
for i in range(0, 30):
    images_training, labels_training = mndata.load_training()
    images_testing, labels_testing = mndata.load_testing()
    classification = Classification(images_training, labels_training, images_testing, labels_testing, sigma=test_sigma)
    classification.create_model()
    with open(write_path, 'a+') as file:
        file.write(str(test_sigma) + ',' + classification.calculate_accuracy(calculate=False, mode='return'))
        file.write('\n')
