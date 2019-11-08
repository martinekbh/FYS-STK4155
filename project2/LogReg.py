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

        beta = np.random.randn(len(X[0]), 1) # why random?

        for epoch in range(n_epochs):   # epoch
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

    '''
    def stocastic_gradient_descent(self, Niter, print_coeffs = False):
        """ SGD with scikit """

        sGD_classifier = SGDClassifier(loss = 'log',
                    penalty = 'l2', max_iter = Niter, eta0 = 0.001, fit_intercept = True,
                    shuffle = True, random_state = self.seed, n_iter_no_change = 5)
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
    '''

    def print_coeff_table(self, beta_coeff, beta_intercept):
        from prettytable import PrettyTable
        table = PrettyTable()
        column_names = ["Predictor", "Coefficient"]
        table.add_column(column_names[0], self.predictors)
        table.add_column(column_names[1], beta_intercept+beta_coeff)
        print(table)

        return

    '''
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
    '''

    def accuracy(self, y, y_pred):
        return np.sum(y == y_pred)/len(y)


if __name__== "__main__":
    # Import breast cancer data
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target
    n = len(y)

    # Create design matrix
    #one_vector = np.ones(np.shape(X[:,0])).reshape(len(X[:,0]),1)
    one_vector = np.ones((n,1))
    A = np.concatenate((one_vector, X), axis = 1)

    seed = 1        # Define seed

    # Set up training data
    X_train, X_test, y_train, y_test = train_test_split(A, y, test_size=0.2, random_state=seed)

    # Scale data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_test  = scaler.transform(X_test)
    X_train = scaler.transform(X_train)

    logreg = LogReg(X_train, y_train, predictor_names = cancer.feature_names)


    l_rate = 0.1
    n_epochs = 100
    beta = logreg.sgd(n_epochs, n_minibatches=30)
    # Martine: Jeg tror vi mangler intercept i beta?

    pred = logreg.predict(X_test)
    acc = logreg.accuracy(y_test, pred)
    print(f"\n\nLogistic regression with {n_epochs} epochs and {l_rate} learning rate")
    print(f"Accuracy: {acc}")
