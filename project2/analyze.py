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
from sklearn.datasets import load_breast_cancer

# Import plotting tools
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter



class data_manager:
    def __init__(self, X, y, predictors = 0):
            # Features and targets
        self.X = X
        self.y = y

        self.n = len(self.y)         # Number of observations
        self.p = np.shape(self.X)[1] # Number of explanatory variables

        # Whacky method to get predictor names correct
        if isinstance(predictors, int):
            self.predictors = np.arange(1,self.p+1)

        else:
            self.predictors = predictors


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

        # Scale data
        X_train = self.X[train_index]
        scaler = StandardScaler()
        scaler.fit(X_train)

        self.X_test  = scaler.transform(self.X[test_index])
        self.X_train = scaler.transform(self.X[train_index])
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


class solver:
    def __init__(self, credit_card_object):
        self.cred = credit_card_object

        self.Xtest, self.Xtrain, self.ytest, self.ytrain = self.cred.split_data()
        self.predictors = self.cred.predictors


    def stocastic_gradient_descent(self, Niter, seed = None, print_coeffs = False):
        if seed != None:
            np.random.seed(seed)

        sGD_classifier = SGDClassifier(loss = 'log',
        penalty = 'l2', max_iter = Niter, eta0 = 0.001,fit_intercept = True ,
        shuffle = True,random_state = seed, n_iter_no_change = 5)
        sGD_classifier.fit(self.Xtrain,self.ytrain.ravel())

        ypred = sGD_classifier.predict(self.Xtest)

        # Scores
        R2_score = self.R2(self.ytest, ypred)
        mSE_score = self.MSE(self.ytest, ypred)
        accuracy = self.accuracy(self.ytest, ypred)
        print(f" MSE = {mSE_score}, R2-score = {R2_score}")
        print(f"accuracy = {accuracy}")


        if print_coeffs == True:
            beta_coeff = sGD_classifier.coef_[0]    # This is a nested list for some reason
            beta_intercept = sGD_classifier.intercept_[0]
            self.print_coeff_table(beta_coeff, beta_intercept)

    def print_coeff_table(self, beta_coeff, beta_intercept):
        from prettytable import PrettyTable
        table = PrettyTable()
        column_names = ["Predictor", "Coefficient"]
        table.add_column(column_names[0], self.predictors)
        table.add_column(column_names[1], beta_intercept+beta_coeff)
        print(table)

        return

    def R2(self, z, z_pred):
        """ Function to evaluate the R2-score """
        z = np.ravel(z)
        z_pred = np.ravel(z_pred)
        mean = (1/len(z))*sum(z)
        r2 = 1 - (sum((z-z_pred)**2)/sum((z - mean)**2))
        return r2

    def MSE(self, z, z_pred):
        """ Function to evaluate the Mean Squared Error """
        z = np.ravel(z)
        z_pred = np.ravel(z_pred)
        mse = (1/len(z))*np.sum((z-z_pred)**2)
        return mse

    def accuracy(self, y, y_pred):
        return np.sum(y == y_pred)/len(y)




if __name__== "__main__":
    #cwd = os.getcwd() # Current working directory
    #filename = cwd + '/default of credit card clients.xls'
    #cred = data_manager("default of credit card clients.xls")

    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target
    analyze_data = data_manager(X,y, predictors = cancer.feature_names)

    integrate = solver(analyze_data)
    integrate.stocastic_gradient_descent(100000, seed = 1, print_coeffs = True)
