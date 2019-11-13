from MNISTModule.classification import Classification
import numpy as np
import plotly as py
import plotly.graph_objects as go


class Graph(Classification):
    def __init__(self,
                 classification=None,
                 granularity=50):
        super().__init__(training_data=classification.training_data,
                         training_labels=classification.training_labels,
                         testing_data=classification.testing_data,
                         testing_labels=classification.testing_labels)
        self.granularity = granularity
        self.classification = classification
        self.classifiers = None

    def data(self, classifiers, surface_name='MADGE Surface', normalized=True):
        # Major surface
        x_space = np.linspace(np.amin(self.training_data, 0)[0], np.amax(self.training_data, 0)[0], self.granularity)
        y_space = np.linspace(np.amin(self.training_data, 0)[1], np.amax(self.training_data, 0)[1], self.granularity)
        X, Y = np.meshgrid(x_space, y_space)
        Z = []
        if normalized:
            multiple_factor = 1
        else:
            multiple_factor = np.divide(
                np.absolute(
                    np.round(
                        np.subtract(
                            np.amax(self.training_data, 0)[1], np.amin(self.training_data, 0)[1]))), 100)
        for x_array, y_array in zip(X, Y):
            z_point = []
            for x_point, y_point in zip(x_array, y_array):
                prediction = self.classification.map_to_point([x_point, y_point], normalized=normalized)
                if prediction is None:
                    z_point.append(0)
                else:
                    z_point.append(np.multiply(prediction, multiple_factor))

            Z.append(z_point)
        Z = np.array(Z)

        # Plot the surface
        trace_surface = go.Surface(x=X, y=Y, z=Z, name=surface_name)

        # Plot classification A
        x_1, y_1, z_1, x_0, y_0, z_0 = [], [], [], [], [], []
        # all_data = np.concatenate((self.training_data, self.testing_data), axis=0)
        # all_data_labels = np.append(self.training_labels, self.testing_labels)
        all_data = self.testing_data
        all_data_labels = self.testing_labels
        self.classifiers = classifiers
        for point, label in zip(all_data, all_data_labels):
            if label == classifiers[0]:
                x_1.append(point[0])
                y_1.append(point[1])
                z_1.append(0)
            elif label == classifiers[1]:
                x_0.append(point[0])
                y_0.append(point[1])
                z_0.append(0)
        trace_scatter_class_a = go.Scatter3d(x=x_1, y=y_1, z=z_1, mode='markers',
                                             name='Classifier {}'.format(classifiers[0]))
        trace_scatter_class_b = go.Scatter3d(x=x_0, y=y_0, z=z_0, mode='markers',
                                             name='Classifier {}'.format(classifiers[1]))
        data = [trace_surface, trace_scatter_class_a, trace_scatter_class_b]
        return data

    def plot_madge_data(self, data, filename, title, normalized=True):
        """
        Generates a plot of the madge data
        :param data: data in the form of go scatter/3d data
        :param filename: name of the output file (will append .html to the end if it is not included)
        :param title: title of the graph
        :return: None
        """
        # Normalized Classifiers
        if normalized:
            multiple_factor = 1
        else:
            multiple_factor = np.divide(
                np.absolute(
                    np.round(
                        np.subtract(
                            np.amax(self.training_data, 0)[1], np.amin(self.training_data, 0)[1]))), 100)
        fig = go.Figure(data=data)
        fig.update_layout(scene=dict(
            xaxis=dict(nticks=4, range=[np.amin(self.training_data, 0)[0], np.amax(self.training_data, 0)[0]],),
            yaxis=dict(nticks=4, range=[np.amin(self.training_data, 0)[1], np.amax(self.training_data, 0)[1]],),
            zaxis=dict(nticks=4, range=[np.multiply(self.classifiers[0], multiple_factor),
                                        np.multiply(self.classifiers[1], multiple_factor)],),
            aspectmode='data'),
            title=title,
            autosize=True,
            margin=dict(l=100, r=50, b=65, t=90))
        py.offline.plot(fig, filename=filename)
