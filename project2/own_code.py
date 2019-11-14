"""
Module that contains all relevant functions of our own making.
"""
# General imports
import numpy as np
from LogReg import LogReg
#from random import random, seed
import os

# Import scikit regression tools
from sklearn.model_selection import train_test_split, cross_validate, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Import plotting tools
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# Where to save the figures and data files
PROJECT_ROOT_DIR = "Results"
FIGURE_ID = "Results/FigureFiles"
DATA_ID = "DataFiles/"

if not os.path.exists(PROJECT_ROOT_DIR):
    os.mkdir(PROJECT_ROOT_DIR)

if not os.path.exists(FIGURE_ID):
    os.makedirs(FIGURE_ID)

if not os.path.exists(DATA_ID):
    os.makedirs(DATA_ID)

def image_path(fig_id):
    return os.path.join(FIGURE_ID, fig_id)

def data_path(dat_id):
    return os.path.join(DATA_ID, dat_id)

def save_fig(fig_id, extension='pdf'):
    plt.savefig(image_path(fig_id) + '.' + extension, format=extension)


def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


def CreateDesignMatrix_X(x, y, d = 2):
    """
    Function for creating a design X-matrix with rows [1, x, y, x^2, xy, xy^2 , etc.]
    Input is x and y mesh or raveled mesh, keyword agruments d is the degree of the polynomial you want to fit.
    """
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((d+1)*(d+2)/2)		# Number of elements in beta
    X = np.ones((N,l))

    for i in range(1,d+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = x**(i-k)*(y**k)

    return X

def SVDinv(A):
    ''' Takes as input a numpy matrix A and returns inv(A) based on singular value decomposition (SVD).
    SVD is numerically more stable than the inversion algorithms provided by
    numpy and scipy.linalg at the cost of being slower.
    '''
    U, s, VT = np.linalg.svd(A)
    #print(U)
    #print(s)
    #print(VT)

    D = np.zeros((len(U),len(VT)))
    for i in range(0,len(VT)):
        D[i,i]=s[i]
    UT = np.transpose(U); V = np.transpose(VT); invD = np.linalg.inv(D)
    return np.matmul(V,np.matmul(invD,UT))


def accuracy(y, pred):
    y = y.ravel()
    pred = pred.ravel()
    return np.sum(y==pred)/len(y)

def MSE(z, z_pred):
    """ Function to evaluate the Mean Squared Error """
    z = np.ravel(z)
    z_pred = np.ravel(z_pred)
    mse = (1/len(z))*np.sum((z-z_pred)**2)
    return mse

def R2(z, z_pred):
    """ Function to evaluate the R2-score """
    z = np.ravel(z)
    z_pred = np.ravel(z_pred)
    mean = (1/len(z))*sum(z)
    r2 = 1 - (sum((z-z_pred)**2)/sum((z - mean)**2))
    return r2

def confidence(beta, X, confidence=1.96):
    weight = np.sqrt( np.diag( np.linalg.inv( X.T @ X ) ) )*confidence
    betamin = beta - weight
    betamax = beta + weight
    return betamin, betamax



def test_train_index(n, test_size=0.33, seed=None):
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


def k_folds(n, k=5, seed = None):
    """
    Returns a list with k lists of indexes to be used for test-sets in
    cross-validation. The indexes range from 0 to n, and are randomly distributed
    to the k groups s.t. the groups do not overlap
    """
    indexes = np.arange(n)

    if seed != None:
        np.random.seed(seed)
    np.random.shuffle(indexes)

    min_size = int(n/k)
    extra_points = n % k

    folds = []
    start_index = 0
    for i in range(k):
        if extra_points > 0:
            test_indexes = indexes[start_index: start_index + min_size + 1]
            extra_points -= 1
            start_index += min_size + 1
        else:
            test_indexes = indexes[start_index: start_index + min_size]
            start_index += min_size
        folds.append(test_indexes)
    return folds

def k_Cross_Validation_logreg(X, y, k=5, epochs=100, n_minibatches=30, seed = None):
    acc_test = []
    acc_train = []

    n = len(y)   # Number of observations
    i = int(n/k) # Size of test set
    test_folds = k_folds(n, k=k, seed=seed)
   
    for indexes in test_folds:
        m = len(indexes)
        X_test = X[indexes]
        y_test = y[indexes]
        X_train = X[np.delete(np.arange(n), indexes)]
        y_train = y[np.delete(np.arange(n), indexes)]
        #X_test = CreateDesignMatrix_X(x_test, y_test, d=d)
        #X_train = CreateDesignMatrix_X(x_train, y_train, d=d)

        logreg = LogReg(X_train, y_train)
        logreg.sgd(n_epochs=epochs, n_minibatches=n_minibatches)
        y_pred = logreg.predict(X_test)
        acc = accuracy(y_test, y_pred)
        train_acc = accuracy(y_train, logreg.predict(X_train))

        acc_test.append(acc)
        acc_train.append(train_acc)
    
    return np.mean(acc_test), np.mean(acc_train)




def k_Cross_Validation(x, y, z, k=5, d=3, reg_method = 'Linear', lmb = None, seed = None):
    """
    Function that performs k-fold cross-validation with Linear, Lasso, or Ridge regression
    on data x, y, z=f(x, y) for some function f that we are trying to model.
        - x, y and z are 1-dimensional arrays.
        - k specifies the number of folds in the cross validation.
        - d specifies the polynomial degree of the linear model.
        - reg_method specifies the regression method (Linear, Lasso, or Ridge)
        - lmb is the lambda parameter for Lasso and Ridge regression.
    """

    error_test = []
    error_train = []
    r2 = []

    n = len(z)              # Number of "observations"
    i = int(n/k)            # Size of test set
    #print(f"\nPERFORMING {k}-FOLD CROSS VALIDATION (with {reg_method} regression):")
    #print(f"Number of observations (n): {n}")
    #print(f"Minimum size of test set: {i}")
    #print(f"Degree of the polynomial: {d}")

    test_folds = k_folds(n, k=k, seed=seed)

    if reg_method == 'Lasso':
        model = Lasso(alpha=lmb, fit_intercept = False, tol=0.001, max_iter=10e6)

    for indexes in test_folds:
        m = len(indexes)
        x_test = x[indexes]
        y_test = y[indexes]
        z_test = z[indexes]
        x_train = x[np.delete(np.arange(n), indexes)]
        y_train = y[np.delete(np.arange(n), indexes)]
        z_train = z[np.delete(np.arange(n), indexes)]

        X_test = CreateDesignMatrix_X(x_test, y_test, d=d)
        X_train = CreateDesignMatrix_X(x_train, y_train, d=d)

        if reg_method == 'Linear':
            beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ z_train
            z_pred_test = X_test @ beta
            z_pred_train = X_train @ beta


        if reg_method == 'Ridge':
            dim = len(X_train.T @ X_train)
            beta = np.linalg.pinv((X_train.T @ X_train) + lmb*np.identity(dim)) @ X_train.T @ z_train
            z_pred_test = X_test @ beta
            z_pred_train = X_train @ beta

        if reg_method == 'Lasso':
            fit = model.fit(X_train, z_train)
            z_pred_train = fit.predict(X_train)
            z_pred_test = fit.predict(X_test)


        error_test.append(sum((z_test - z_pred_test)**2)/len(z_test))
        error_train.append(sum((z_train - z_pred_train)**2)/len(z_train))
        #error_test.append(((z_test - z_pred_test)@(z_test - z_pred_test).T)/len(z_pred))
        #error_train.append(((z_train - z_pred_train)@(z_train - z_pred_train).T)/len(z_pred))
        r2.append(R2(z_test, z_pred_test))
    
    test_err = np.mean(error_test)
    train_err = np.mean(error_train)
    r2_score = np.mean(r2)

    return test_err, train_err, r2_score



# Function for making plot
def make3Dplot(x, y , z, title=None, name=None, size=(16,12),  show=True):
    """
    Function for making 3D plot of a function f(x,y)=z. 
    x, y, and z must be np.meshgrid, or one dimentional arrays.
    """
    fig = plt.figure(figsize=size)
    ax = fig.gca(projection='3d')

    if len(z.shape) == 1: # If z is not meshgrid
        surf = ax.plot_trisurf(x, y, z, cmap=cm.coolwarm, linewidth = 0,
                            antialiased=False)
    else:
        surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")

    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    if title:
        plt.title(title)
    if name:
        save_fig(name)
    if show:
        plt.show()
    else:
        plt.close()



def make3Dplot_multiple(x, y, zlist, title=None, name=None, size=(25,10), show=True):
    """
    Function for making a plot with multiple 3D subplots, where each subplot 
    has the same values for x and y, but the z=f(x,y) values differ.
    x and y must be np.meshgrid.
    zlist is a list of np.meshgrids.
    """
    fig = plt.figure(figsize=size)

    for i in range(len(zlist)):
        ax = fig.add_subplot(1,len(zlist),i+1, projection='3d')       # Create subplot
        #fig.add_axes([0.5 , 0.1, 0.4, 0.8])
        #ax = fig.gca(projection='3d')       # Get axes of current subplot
        surf = ax.plot_surface(x, y, zlist[i], cmap=cm.coolwarm, linewidth=0, antialiased=False) # The surface


        # Customize the z axis.
        ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))


        plt.xlabel("x-axis")
        plt.ylabel("y-axis")

        if title:
            plt.title(title[i])
    
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

    if name:
        save_fig(name)
    if show:
        plt.show()
    else:
        plt.close()


