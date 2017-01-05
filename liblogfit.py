# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 21:21:17 2016
@author: Marco Bressan
"""

import numpy as np


def z(x):
    """Evaluate the sigmoid function at x."""
    return 1.0/(1.0 + np.exp(-x))


def h(Theta, X):
    """Evaluate the sigmoid function at each element of <Theta,X>."""
    return np.array([z(np.dot(Theta, x)) for x in X])


def gradient(Theta, X, Y):
    """Compute the gradient of the log-likelihood of the sigmoid."""
    pX = h(Theta, X) # i.e. [h(x) for each row x of X]
    return np.dot((Y - pX), X)


def logfit(X, Y, alpha=1, itr=10):
    """Perform a logistic regression via gradient ascent."""
    Theta = np.zeros(X.shape[1])
    for i in range(itr):
        Theta += alpha * gradient(Theta, X, Y)
    return Theta


def normalize(X):
    """Normalize an array, or a dataframe, to have mean 0 and stddev 1."""
    return (X - np.mean(X, axis=0))/(np.std(X, axis=0))


def tprfpr(P, Y):
    """Return the False Positive Rate and True Positive Rate vectors of the given classifier."""
    Ysort = Y[np.argsort(P)[::-1]]
    ys = np.sum(Y)
    tpr = np.cumsum(Ysort)/ys # [0, 0, 1, 2, 2, 3,..]/18
    fpr = np.cumsum(1-Ysort)/(len(Y)-ys)
    return (tpr, fpr)


def auc(fpr, tpr):
    """Compute the Area Under the Curve (AUC) given vectors of
    false positive rate and true positive rate"""
    return(np.diff(tpr) * (1 - fpr[:-1])).sum()


