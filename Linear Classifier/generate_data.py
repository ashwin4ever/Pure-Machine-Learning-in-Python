'''
CS 5793 AI 2

Basic Classifiers

Generate Data points

'''

import numpy as np


def generateGaussData(mean , cov , dim):
      a , b = np.random.multivariate_normal(mean, cov, dim).T
      return (a , b)
