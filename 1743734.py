# -*- coding: utf-8 -*-
"""
@author: Fernando Crema, based on code by Marco Bressan
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def z(x):
    """
    Evaluate the sigmoid function at x.
    :param x: Vector.
    :return: Value returned.
    """
    return 1.0/(1.0 + np.exp(-x))


def h(Theta, X):
    """
    Evaluate the sigmoid function at each element of <Theta,X>.
    :param Theta: Coefficients of the model.
    :param X: Vector of features.
    :return: The sigmoid function evaluated in each element.
    """
    return np.array([z(np.dot(Theta, x)) for x in X])


def gradient(Theta, X, Y):
    """
    Compute the gradient of the log-likelihood of the sigmoid
    :param Theta: Coefficients.
    :param X: Matrix of features.
    :param Y: Vector Y.
    :return: Return
    """
    pX = h(Theta, X) # i.e. [h(x) for each row x of X]
    return np.dot((Y - pX), X)


def logfit(X, Y, alpha=1, itr=10):
    """
    Perform a logistic regression via gradient ascent.
    :param X: The data matrix.
    :param Y: The class.
    :param alpha: Coefficient to control the gradient.
    :param itr: Number of iterations.
    :return: The model coefficients.
    """
    Theta = np.zeros(X.shape[1])
    for i in range(itr):
        Theta += alpha * gradient(Theta, X, Y)
    return Theta


def normalize(X):
    """
    Normalize an array, or a dataframe, to have mean 0 and stddev 1.
    :param X: Array or dataframe.
    :return: The Array normalized.
    """
    return (X - np.mean(X, axis=0))/(np.std(X, axis=0))


def tprfpr(P, Y):
    """
    Return the False Positive Rate and True Positive Rate vectors of the given classifier.
    :param P: The vector of probabilities.
    :param Y: The real classes.
    :return: the true positive rate and the false positive rate
    """
    Ysort = Y[np.argsort(P)[::-1]]
    ys = np.sum(Y)
    tpr = np.cumsum(Ysort)/ys # [0, 0, 1, 2, 2, 3,..]/18
    fpr = np.cumsum(1-Ysort)/(len(Y)-ys)
    return tpr, fpr


def auc(fpr, tpr):
    """
    Compute the Area Under the Curve (AUC) given vectors of false positive rate and true positive rate
    :param fpr: The false positice rate.
    :param tpr: The true positive rate.
    :return: The area under the curve (AUC)
    """
    return (np.diff(tpr) * (1 - fpr[:-1])).sum()


def preprocessing(train, test):
    """
    Preprocess data to eliminate NAN.
    :param train: The train set data.
    :param test: The test set data.
    :return: The data preprocessed.
    """
    train.dropna(inplace=True)
    test.dropna(inplace=True)

    return train.reset_index(drop=True), test.reset_index(drop=True)


def gentrain(train):
    """
    Generates the train set with the 1 column and the y vector.
    :param train: The matrix of training.
    :return: The matrix x with the 1 column and the y vector.
    """

    m, p = train.shape
    x = np.ones((m, p))
    x[:, 1:] = train[:, :-1]

    return x, train[:, -1]


def gentest(test):
    """
    Generates the test set matrix.
    :param test: The matrix of testing.
    :return: Matrix with the 1 column.
    """
    m, p = test.shape
    x = np.ones((m, p+1))
    x[:, 1:] = test

    return x


def genplot(fpr, tpr):
    """
    Plot the ROC curve given fpr and tpr vectors.
    :param fpr: False positive rate vector.
    :param tpr: True positive rate vector.
    """
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title('ROC Curve')
    plt.plot(fpr, tpr)
    plt.grid()
    plt.savefig("1743734-roc.png")


def main():
    np.set_printoptions(precision=2)
    # Read file names.
    train, test = sys.argv[1:]

    # Read files with pandas read_csv
    train, test = pd.read_csv(train), pd.read_csv(test)

    # Preprocess train and test. eliminate NAN
    train, test = preprocessing(train=train, test=test)
    x, y = gentrain(train.as_matrix(columns= list(train.columns.values)))

    # 1. the vector theta of parameter estimates obtained through the logistic regression
    theta = logfit(X=x, Y=y, alpha=0.001, itr=100)
    print(theta)

    # 2. the ROC curve obtained on the training dataset, on a file called ID-roc.png, in PNG format

    # Predicted probabilities
    p = h(Theta=theta, X=x)

    # Find true positive rate and false positive rate.
    tpr, fpr = tprfpr(P=p, Y=y)

    # Plot the ROC curve.
    genplot(fpr=fpr, tpr=tpr)

    # 3. the AUC value obtained on the training dataset
    print(auc(fpr=fpr, tpr=tpr))

    # 4. the scores predicted for the observations in the test dataset
    print(h(Theta=theta, X=gentest(test)))

if __name__ == "__main__":
    main()

