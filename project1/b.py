'''
Part b) of project 1
'''
# Import regression tools
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from random import random, seed

# Import plotting tools
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# Make data
x = np.arange(0, 1, 0.2)   # Numbers on x-axis
y = np.arange(0, 1, 0.2)   # Numbers on y-axis
k = len(x)                  # Number of "observations" on each axis  
x, y = np.meshgrid(x,y)     # Create meshgrid of x and y axes

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

z = FrankeFunction(x, y)    # Our z-values

# Function for making plot
def make3Dplot(x, y, z, title=None, name=None, show=True):
    """
    Function for making 3D plot. x, y, and z must be np.meshgrid
    """
    fig = plt.figure(figsize=(16,12))
    ax = fig.gca(projection='3d')

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
        plt.savefig(name)
    if show:
        plt.show()
    else:
        plt.close()

def CreateDesignMatrix_X(x, y, n = 2):
	"""
	Function for creating a design X-matrix with rows [1, x, y, x^2, xy, xy^2 , etc.]
	Input is x and y mesh or raveled mesh, keyword agruments n is the degree of the polynomial you want to fit.
	"""
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = x**(i-k)*(y**k)

	return X

x1 = np.ravel(x)        # Convert matrix (meshgrid) to vector
y1 = np.ravel(y)        # Convert matrix (meshgrid) to vector
z1 = np.ravel(z)        # Convert matrix (meshgrid) to vector
n = len(x1)             # Number of "observations"

#xy = np.c_[x1,y1]       # Rearrange x and y into (x,y) tuples

X = CreateDesignMatrix_X(x1, y1, n=2)

"""
# MOVED THIS PART TO a.py

# Do OLS Linear Regression on whole data-set to compare with the results in part a)
linreg = LinearRegression(fit_intercept=False)
linreg.fit(X, z1)
p = len(linreg.coef_)

for i in range(p):
    print(f"Beta_{i:2d} = {linreg.coef_[i]:12.8f}")


z_pred = linreg.predict(X)
r2 = r2_score(z1, z_pred)
mse = mean_squared_error(z1, z_pred)
print(f"Mean Squared Error: {mse}")
print(f"R2-score: {r2}")

k = int(np.sqrt(len(z_pred)))
make3Dplot(x,y,z_pred.reshape(k,k), title="Linear regression with scikit", 
            name="lin_reg_scikit_franke_func.png", show=False)
"""


X_train, X_test, z_train, z_test = train_test_split(X,z1, test_size=0.2, random_state=1)

#print(f"\nX\n: {X}")
#print(f"\nX_test\n: {X_test}")
#print(f"\nz_test\n: {z_test}")

# CROSS VALIDATION:

def split_data(X, z, k=5):
    """
    Splits data into k groups of equal size
    """
    #X_groups = np.split(X)     # This splits matrix, but not randomly
    #z_groups = np.split(z)     # This splits z, but not randomly

    X_test_sets = []
    z_test_sets = []

    X_pool = X
    z_pool = z
    test_size = int(len(z)/k)

    for i in range(k-1):
        X_train, X_test, z_train, z_test = train_test_split(X_pool, z_pool, test_size=test_size)
        X_test_sets.append(X_test)
        z_test_sets.append(z_test)
        X_pool = X_train
        z_pool = z_train

    X_test_sets.append(X_pool) # append last set
    z_test_sets.append(z_pool)

    return X_test_sets, z_test_sets


def MSE(z, z_pred):
    """ Function to evaluate the Mean Squared Error """
    z = np.ravel(z)
    z_pred = np.ravel(z_pred)
    mse = (1/len(z))*sum((z-z_pred)**2)
    return mse

def R2(z, z_pred):
    """ Function to evaluate the R2-score """
    z = np.ravel(z)
    z_pred = np.ravel(z_pred)
    mean = (1/len(z))*sum(z)
    r2 = 1 - (sum((z-z_pred)**2)/sum((z - mean)**2))
    return r2


def k_Cross_Validation(X, z, k=5):
    """
    Function that performs k-fold cross validation.
    X is the design matrix, z is the result variable.
    """
    print(f"\nPERFORMING {k}-FOLD CROSS VALIDATION:\n")
    n = len(z)                                          # Number of "observations"
    i = int(n/k)                                        # Size of test set
    print(f"Number of observations (n): {n}")
    print(f"Minimum size of test set: {i}")
    X_groups, z_groups = split_data(X, z, k)            # Split data into k groups

    # Check that the groups are correct
    for i in range(len(z_groups)):
        print(f"\nGruppe {i}:")
        print(f"[1, x, y, x^2, xy, y^2] \t\t z-value \t\t FrankeFunction(x,y)")
        for j in range(len(z_groups[i])):
            f = FrankeFunction(X_groups[i][j][1], X_groups[i][j][2])        #test z values
            print(f"{X_groups[i][j]}\t{z_groups[i][j]}\t{f}")

    mse = []    # List to store the MSEs
    r2 = []     # List to store the R2-scores

    # Do cross validation
    for i in range(len(z_groups)):
        print(f"\nLinear regression with CV test-group nr. {i}:")

        X_test = X_groups[i]
        z_test = z_groups[i]
        p = len(X_test[1])      # columns in the X matrix i.e. the number of betas

        #X_train = np.delete(X_groups, i, axis=0)

        # Create design matrix for training set:
        X_train = np.zeros((0,p))
        for group in np.delete(X_groups, i, axis=0):
            X_train = np.concatenate((X_train, group), axis=0)

        # The z-values in the training set:
        z_train = z_groups[:i] + z_groups[i+1:]     # Pick groups to be in training set
        z_train = np.concatenate(z_train, axis=0)   # Merge z-values into one array

        print(f"Number of observations in training set: {len(z_train)}")
        print(f"Dimentions of design matrix for training set: {len(X_train[0])}x{len(X_train)}")

        P = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T)
        beta = P.dot(z_train)
        print(f"beta = {beta}")

        z_pred = X_test @ beta              # Test model on test data
        mse.append(MSE(z_test, z_pred))     # MSE
        r2.append(R2(z_test, z_pred))       # R2-score
        print(f"MSE = {mse[i]}\nR2 = {r2[i]}")

    
    return

k_Cross_Validation(X, z1, k=2)    


#poly = PolynomialFeatures(degree = 5)

# CROSS VALIDATION USING SCIKIT?









