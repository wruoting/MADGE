from classification_set_n import ClassificationSetN
from n_point import NPoint
import gzip
import numpy as np
from mnist import MNIST
import ast

a = NPoint((1, 2, 3), type=1)
b = NPoint((1, 2, 4), type=0)
d = NPoint((1, 2, 5), type=1)
c = ClassificationSetN()
# c.add_point(a)
# c.add_point(b)

# print(c.calculate_madge_data_and_map_to_point(d))


# path = './SampleData/MNIST/train-images-idx3-ubyte.gz'
samples_path = './SampleData/MNIST'
# f = gzip.open(path, 'r')
#
# image_size = 28
# num_images = 5
# #https://stackoverflow.com/questions/40427435/extract-images-from-idx3-ubyte-file-or-gzip-via-python
# f.read(1)
# buf = f.read(image_size * image_size * num_images)
# data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
# print(data.shape)
# data = data.reshape(num_images, image_size, image_size, 1)

mndata = MNIST(samples_path)
images_training, labels_training = mndata.load_training()
images_testing, labels_testing = mndata.load_testing()

mnist_training_set = ClassificationSetN()

# We are going to calculate a sigma here that is proportional to the average of the range of the points
# This calculation will allow us to reflect gaussian area with a relative rather than absolute distance using
# any arbitrary sigma.
# 1. Take the length of the training/testing labels. We can shortcut and take the first one
#    We can then make an empty array of zeros with that as the distance
train_label_dim = len(images_training[0])
train_label_sigma_max = np.zeros(train_label_dim)
train_label_sigma_min = np.zeros(train_label_dim)

for image, label in zip(images_training, labels_training):
    mnist_training_set.add_point(NPoint(image, type=label))
    # 2. Each point needs to look for a max and a min for sigmas. Runtime unfortunately N^2
    for element, index in enumerate(image):
        if element > train_label_sigma_max[index]:
            train_label_sigma_max[index] = element
        if element < train_label_sigma_min:
            train_label_sigma_min[index] = element

# 3. Calculate the average sigma and use this ubiquitously (a + b +c .... /n)

for image, label in zip(images_testing, labels_testing):
    print('Real')
    print(label)
    print('Test')
    print(mnist_training_set.calculate_madge_data_and_map_to_point(NPoint(image, type=label)))

# import matplotlib.pyplot as plt
# image = np.asarray(data[2]).squeeze()
# plt.imshow(image)
# plt.show()