from MNISTModule.classification_set import ClassificationSet
from MNISTModule.classification import Classification
from MNISTModule.graph import Graph
from MNISTModule.madge_calculator import read_data_from_file
import plygdata as pg
from IrisModule.sanitize_iris_data import create_iris_data
from mnist.loader import MNIST

# samples_path = './SampleData/MNIST'
# 
# mndata = MNIST(samples_path)
# images_training, labels_training = mndata.load_training()
# images_testing, labels_testing = mndata.load_testing()
# 
# mnist_training_set = ClassificationSet()
# 
# classification = Classification(images_training, labels_training, images_testing, labels_testing)
# 
# classification.calculate_accuracy()

# x_dim = "1"
# y_dim = "10000"
# validation_data_ratio = 0.2
# path = "./SampleData/SpiralStretch/ClassifySpiralDataNonSquare-{}-{}.txt".format(x_dim, y_dim)
# training_data_set = read_data_from_file(path)
# # This is the split data we will be using to generate all our graphs
# X_train, Y_train, X_validate, Y_validate = pg.split_data(training_data_set, validation_size=validation_data_ratio)
# 
# 
# classification = Classification(X_train, Y_train, X_validate, Y_validate)
# 
# classification.calculate_accuracy()


# Gauss Data
# validation_data_ratio = 0.2
# path = "./SampleData/SpiralStretch/ClassifySpiralDataNonSquare-1-20.txt"
# training_data_set = read_data_from_file(path)
# # This is the split data we will be using to generate all our graphs
# X_train, Y_train, X_validate, Y_validate = pg.split_data(training_data_set, validation_size=validation_data_ratio)
#
#
# classification = Classification(X_train, Y_train, X_validate, Y_validate)
#
# classification.calculate_accuracy(mode='prod')
# graph = Graph(classification)
# graph.plot_madge_data(graph.data((-1, 1), normalized=False),
#                       filename='11-11-19-Classification.html',
#                       title='Classification',
#                       normalized=False)
# graph_normalized = Graph(classification)
# graph_normalized.training_data = classification.normalized_training_data
# graph_normalized.testing_data = classification.normalized_testing_data
# graph_normalized.plot_madge_data(graph_normalized.data((-1, 1)),
#                                  filename='11-11-19-Normalized-Classification.html',
#                                  title='Normalized-Classification')


## Iris Classification
X_train, Y_train, X_validate, Y_validate = create_iris_data(validation_data_ratio=0.2)