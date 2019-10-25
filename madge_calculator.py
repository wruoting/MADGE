from scipy.spatial import distance
from scipy import special as sp
from scipy.stats import multivariate_normal
import autograd.numpy as agnp


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


def classify_by_distance(classifiers, weight):
    """
    :param classifiers: list of classifiers, array like
    :param weight: float of the weight predicted
    :return: the closest classifier to the weight
    """
    sorted_classifiers = agnp.sort(classifiers, axis=None)
    classifier = None
    classify_index = None
    if weight <= sorted_classifiers[0]:
        classifier = sorted_classifiers[0]
    elif weight >= sorted_classifiers[-1]:
        classifier = sorted_classifiers[-1]
    if classifier is None:
        for index, value in enumerate(sorted_classifiers[1:]):
            # We know it's in between this index and the lower index
            if weight <= value:
                classify_index = index
    if classify_index is not None:
        classifier = sorted_classifiers[classify_index] \
            if weight - sorted_classifiers[classify_index-1] > sorted_classifiers[classify_index] - weight \
            else sorted_classifiers[classify_index-1]
    return classifier

# https://math.stackexchange.com/questions/2854262/circle-shaped-integration-of-a-bivariate-normal-distribution
# http://socr.umich.edu/HTML5/BivariateNormal/
# print(gaussian_area(1.96, 0, 1))


