"""
Module that contains all relevant functions of our own making.
"""
# General imports
import numpy as np
import pandas as pd
#from random import random, seed
import os

# Import scikit regression tools
from sklearn.model_selection import train_test_split, cross_validate, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge, SGDClassifier
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures,StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer

# Import plotting tools
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter



class credit_data:
    def __init__(self, credit_card_filename):
        nanDict = {}

        df = pd.read_excel(filename, header=1, skiprows=0, index_col=0, na_values=nanDict)
        df.rename(index=str, columns={"default payment next month": "defaultPaymentNextMonth"}, inplace=True)

            # Features and targets
        self.X = df.loc[:, df.columns != 'defaultPaymentNextMonth'].values
        self.y = df.loc[:, df.columns == 'defaultPaymentNextMonth'].values

        self.n = len(self.y)         # Number of observations
        self.p = np.shape(self.X)[1] # Number of explanatory variables


    def split_data(self, Testsize = 0.33, seed = None):
        """
            input:
                test_size: The size of the test data which is going to be
                validated against train data with size 1 - test_size.
                Default size 1/5.
                seed:

            return:
                test_data: Ndarrray
                train_data: Ndarray

        """
        """Trainsize = 1 - Testsize
        print(np.shape(self.X), np.shape(self.y))

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = Testsize, random_state = seed)

        self.Xtrain = X_train
        self.Xtest = X_test
        self.ytrain = y_train
        self.ytest = y_test

        """
        test_index, train_index = self.test_train_index(self.n)

        self.X_test  = self.X[test_index]
        self.X_train = self.X[train_index]
        self.y_train = self.y[train_index]
        self.y_test  = self.y[test_index]

        return self.X_test, self.X_train, self.y_test, self.y_train



    def test_train_index(self, n, test_size=0.33, seed=None):
        """
        Returns indexes for randomly selected train set and test set.
        args:
            n is an int specifying the number of indexes
            test_size is between 0 and 1, and is the size of the test set (as a ratio)
        """
        indexes = np.arange(n)

        if seed != None:
            np.random.seed(seed)
        np.random.shuffle(indexes)

        n_test = int(n*test_size)
        test_indexes = indexes[0:n_test]
        train_indexes = indexes[n_test:]

        return test_indexes, train_indexes



#class Logreg(credit_data):
#    def logistic_regression(self):

class solver:
    def __init__(self, credit_card_object):
        self.cred = credit_card_object
        self.Xtest, self.Xtrain, self.ytest, self.ytrain = self.cred.split_data()


    def stocastic_gradient_descent(self, Niter):

        sGD_classifier = SGDClassifier(loss = 'log',
        penalty = 'l2', max_iter = Niter, eta0 = 0.0, shuffle = True, n_iter_no_change = 10)
        sGD_classifier.fit(self.Xtrain,self.ytrain.ravel())

        return sGD_classifier.get_params()





if __name__== "__main__":
    cwd = os.getcwd() # Current working directory
    filename = cwd + '/default of credit card clients.xls'
    cred = credit_data("default of credit card clients.xls")


    integrate = solver(cred)
    print(integrate.stocastic_gradient_descent(1000))
