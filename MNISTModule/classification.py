import numpy as np
from MNISTModule.madge_calculator import replace_with_one, classify_by_distance
from MNISTModule.n_point import NPoint
from MNISTModule.classification_set import ClassificationSet


class Classification(object):
    def __init__(self,
                 training_data,
                 training_labels,
                 testing_data,
                 testing_labels,
                 sigma=0.02):

        self.classification_set = ClassificationSet()
        self.training_data = training_data
        self.training_labels = training_labels
        self.testing_data = testing_data
        self.testing_labels = testing_labels
        self.normalized_training_data = []
        self.normalized_training_labels = training_labels
        self.normalized_testing_data = []
        self.normalized_testing_labels = []
        self.range_vector = None
        self.sigma = sigma
        
    def calculate_accuracy(self, mode='test'):
        # 1. Each point needs to look for a max and a min for sigmas. Runtime unfortunately N*M
        print('Processing images')
        # Figure out what the range is, and then shrink to normalize them

        replace_with_one_vector = np.vectorize(replace_with_one)
        self.range_vector = replace_with_one_vector(np.subtract(
            np.amax(self.training_data, 0), np.amin(self.training_data, 0)))
        images_training_normalize = np.divide(self.training_data, self.range_vector)
        images_testing_normalize = np.divide(self.testing_data, self.range_vector)
        self.normalized_training_data = images_training_normalize
        self.normalized_testing_data = images_testing_normalize
        # We are going to calculate a sigma here that is proportional to the average of the range of the points
        # This calculation will allow us to reflect gaussian area with a relative rather than absolute distance using
        # any arbitrary sigma.
        # 2. Take the length of the training/testing labels. We can shortcut and take the first one
        #    We can then make an empty array of zeros with that as the distance

        for image, label in zip(images_training_normalize, self.training_labels):
            self.classification_set.add_point(NPoint(image, type=label))
        # 3. Calculate the average sigma and use this ubiquitously
        # We are going to use the equation sum(n_i/sum(n) * range(w_i)/6),
        # where n_i is the ith dimension, sum(n) is the sum of the range of all dimensions
        # w is the range of the dimension at i

        match = 0
        index_testing = 0
        # sigma = 0.9
        print('Running Testing Data')
        for image, label in zip(images_testing_normalize, self.testing_labels):
            classification = classify_by_distance(
                    np.unique(self.training_labels),
                    self.classification_set.calculate_madge_data_and_map_to_point(NPoint(image, type=label), self.sigma))
            self.normalized_testing_labels.append(classification)
            if label == classification:
                match = match + 1
            index_testing = index_testing + 1
            if mode == 'test':
                print('Processing {} out of {}'.format(index_testing, 100))
                print("Test")
                print(label)
                print('Real')
                print(classification)
                print('---------------')
                if index_testing == 100:
                    break

        with open('Accuracy.txt', "w+") as f:
            f.write(str(np.divide(match, index_testing)))

    def map_to_point(self, point, normalized=True):
        if not normalized:
            normalized_point = np.divide(point, self.range_vector)
        else:
            normalized_point = point
        return classify_by_distance(
            np.unique(self.training_labels),
            self.classification_set.calculate_madge_data_and_map_to_point(NPoint(normalized_point), self.sigma))

