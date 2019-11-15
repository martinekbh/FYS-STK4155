"""
Module that contains all relevant functions of our own making.
"""
# General imports
import numpy as np
#from random import random, seed
import os

# Import scikit regression tools
from sklearn.model_selection import train_test_split, cross_validate, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge, SGDClassifier
from sklearn.preprocessing import PolynomialFeatures,StandardScaler, OneHotEncoder
from sklearn.datasets import load_breast_cancer


class LogReg:
    def __init__(self, X, y, predictor_names = None, seed = None):
        self.X = X
        self.y = y
        self.seed = seed
        if seed != None:
            np.random.seed(seed)

        self.predictors = predictor_names  # Names of the predictors
        if isinstance(self.predictors, np.ndarray):
            self.print_coeffs=True
        else:
            self.print_coeffs=False

    def sgd(self, n_epochs, n_minibatches=None):
        """ Own sgd """
        X = self.X
        y = self.y
        n = len(X)

        if n_minibatches == None: # Default number of minibatches is n
            n_minibatches = n
            batch_size = 1

        else:
            batch_size = int(n / n_minibatches)

        beta = np.random.randn(len(X[0]), 1)

        for epoch in range(n_epochs):               # epoch
            for i in range(n_minibatches):          # minibatches
                random_index = np.random.randint(n_minibatches)

                xi = X[random_index * batch_size: random_index*batch_size + batch_size]
                yi = y[random_index * batch_size: random_index*batch_size + batch_size]
                yi = yi.reshape((batch_size, 1))

                p = 1/(1 + np.exp(-xi @ beta))
                gradient = -xi.T @ (yi - p) 
                l = self.learning_schedule(epoch*n_minibatches + i)
                beta = beta - l * gradient
                self.beta = beta

        self.beta = beta

        if self.print_coeffs == True:
            self.print_coeff_table(beta[0], beta[1:])

        return beta

    def learning_schedule(self, t, t0=5, t1=50):
        return t0/(t+t1)

    def predict(self, x, beta=None):
        if beta == None:
            beta = self.beta
        pred = np.round(1/(1 + np.exp(-x@beta))).ravel()
        return pred

    def print_coeff_table(self, beta_intercept, beta_coeff):
        from prettytable import PrettyTable
        table = PrettyTable()
        intercept = np.array(["intercept"])
        predictors = np.concatenate((intercept, self.predictors))
        values = np.concatenate((np.array([beta_intercept]), beta_coeff)).ravel()
        column_names = ["Predictor", "Coefficient"]
        table.add_column(column_names[0], predictors)
        table.add_column(column_names[1], values)
        print(table)
        return

    def accuracy(self, y, y_pred):
        return np.sum(y == y_pred)/len(y)


