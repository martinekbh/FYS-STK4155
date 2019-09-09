'''
Part a) of project 1
'''
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed

# Make data
x = np.arange(0, 1, 0.05)   # Numbers on x-axis
y = np.arange(0, 1, 0.05)   # Numbers on y-axis
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

# Plot the surface of the Franke function
make3Dplot(x,y,z, title="Franke Function", name="franke_func.png")


# OLS REGRESSION: fit a polynomial model with x and y up to 5-th order:

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

# Transform (meshgrid) from matrices to vectors
x_1 = np.ravel(x)
y_1 = np.ravel(y)
z_1 = np.ravel(z)   #+ np.random.random(n)*1 # Add noise if wanted
n = len(z_1)        # Number of observations (n=k*k, see beginning of program)

# Create the design matrix
X = CreateDesignMatrix_X(x_1,y_1,n=5)   # n is the degree of the model polynomial 

# Find beta and predict z-values
beta = np.linalg.inv((X.T.dot(X))).dot(X.T).dot(z_1)        # MAYBE USE np.linalg.pinv INSTEAD?
p = len(beta)                       # p is the complesity of the model
z_pred = X @ beta                   # Predicted z values
z_pred = z_pred.reshape(k,k)        # Make z into appropriate matrix so we can plot it

# Plot the surface of the predicted values
make3Dplot(x,y,z_pred, title="Linear regression (Franke function)", 
                name ="lin_reg_franke_func.png", show=True)

# SCIKIT (do we want to use it?):
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



# Find confidence intervals of the betas by computing the variances.
s2 = (np.linalg.norm(z) - np.linalg.norm(z_pred))**2/(n - p)
variance_beta = s2*np.linalg.inv(X.T.dot(X))    # Covariance matrix
beta_var = np.diag(variance_beta)               # Variances of the betas

# NB: Someone check that the variances are correct, and make CI's for the betas.



# Evaluate MSE and R^2-score
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

print(f"Mean Squares Error: {MSE(z, z_pred):.5f}")      # NB: Someone check that this value is correct...
print(f"R2-score: {R2(z, z_pred):.5f}")                 # NB: Someone check that this value is correct...




