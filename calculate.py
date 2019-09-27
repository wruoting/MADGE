from classification_set_n import ClassificationSetN
from n_point import NPoint
from numpy import linalg as LA
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

index_training = 0
for image, label in zip(images_training, labels_training):
    mnist_training_set.add_point(NPoint(image, type=label))
    # 2. Each point needs to look for a max and a min for sigmas. Runtime unfortunately N^2
    print('Processing {} out of {}'.format(index_training, len(images_training)))
    index_training = index_training + 1
    for element, index in enumerate(image):
        if element > train_label_sigma_max[index]:
            train_label_sigma_max[index] = element
        if element < train_label_sigma_min[index]:
            train_label_sigma_min[index] = element
# 3. Calculate the average sigma and use this ubiquitously
# We can just do an l1 of the set
train_label_sigma = LA.norm(np.subtract(train_label_sigma_max, train_label_sigma_min))
mnist_training_set.sigma = np.divide(train_label_sigma, 50)# idk thirty sigma?
print('Sigma value: {}'.format(train_label_sigma))

match = 0
total = len(images_testing)
index_testing = 0
print('Running Testing Data')
for image, label in zip(images_testing, labels_testing):
    if label == round(mnist_training_set.calculate_madge_data_and_map_to_point(NPoint(image, type=label))):
        match = match + 1
    index_testing = index_testing + 1
    print('Processing {} out of {}'.format(index_testing, 100))
    print("Test")
    print(label)
    print('Real')
    print(mnist_training_set.calculate_madge_data_and_map_to_point(NPoint(image, type=label)))
    if index_testing == 100:
        break

with open('Accuracy.txt', "w+") as f:
    f.write(str(np.divide(match, index_testing)))

# import matplotlib.pyplot as plt
# image = np.asarray(data[2]).squeeze()
# plt.imshow(image)
# plt.show()