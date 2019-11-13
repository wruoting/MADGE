from scipy.spatial import distance
from scipy import special as sp
from scipy.stats import multivariate_normal
import autograd.numpy as agnp
import ast

def euclidean(a, b):
    """
    :param a: point a array_like
    :param b: point b array_like
    :return:
    """
    return distance.euclidean(a, b)


def gaussian_area(x, mean, sigma):
    """
    :param x: lower/higher bound
    :param mean: gaussian param mean
    :param sigma: gaussian param sigma
    :return: area under curve from x -> inf or x-> -inf
    """
    double_prob = agnp.abs(sp.erf((x - mean) / (sigma * agnp.sqrt(2))))
    p_zero_to_bound = agnp.divide(double_prob, 2)
    return agnp.subtract(0.5, p_zero_to_bound)


def multi_variate_gaussian_area(xy, mean, sigma):
    """
    :param xy: lower/higher bound, array like [x,y]
    :param mean: gaussian param mean, array like [a,b]
    :param sigma: gaussian param sigma, array like [[a1, b1],[a2, b2]]. a normal cov is [[1,0],[0,1]]
    :return:
    """
    multivariate_normal_fx = multivariate_normal(mean=mean, cov=sigma)
    return multivariate_normal_fx.pdf(xy)


def replace_with_one(a):
    if a == 0:
        return 1
    return a


def classify_by_distance(classifiers, weight):
    """
    :param classifiers: list of classifiers, array like
    :param weight: float of the weight predicted
    :return: the closest classifier to the weight
    """
    sorted_classifiers = agnp.sort(classifiers, axis=None)
    if weight <= sorted_classifiers[0]:
        return sorted_classifiers[0]
    if weight >= sorted_classifiers[-1]:
        return sorted_classifiers[-1]
    for index, classifier in enumerate(sorted_classifiers[0:-1]):
        final_classifier = distance_between_two(classifier, sorted_classifiers[index+1], weight)
        if final_classifier is not None:
            return final_classifier
    return None


def distance_between_two(bot, top, weight):
    if weight < bot or weight > top:
        return None
    if weight == bot:
        return bot
    if weight == top:
        return top
    if weight - bot > top - weight:
        return top
    if top - weight > weight - bot:
        return bot
    else:
        print('Cant make accurate prediction')
        return None
    

def read_data_from_file(path):
    """
    Read data from a file and returns it as a numpy array
    :param path: path of file relative to this file
    :return: numpy array
    """
    f = open(path, "r")
    return agnp.array(ast.literal_eval(f.readlines()[0]))


def convert_array_to_array_of_tuples(input_array):
    """
    Converts each array of an array into tuples
    :param input_array: path of file relative to this file
    :return: array of tuples
    """
    to_tuple = lambda v: tuple(v)
    return [to_tuple(ai) for ai in input_array]