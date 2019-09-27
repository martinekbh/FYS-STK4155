'''
Part e) of project 1
'''


# Import regression tools
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
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

X = CreateDesignMatrix_X(x1, y1, n=5)



X_train, X_test, z_train, z_test = train_test_split(X,z1, test_size=0.2, random_state=1)

#Scikits lasso with lambda/alpha value of our choice
clf = linear_model.Lasso(alpha=0.2)
lasso= clf.fit(X_train, z_train)

z_pred = lasso.predict(X_test)
r2 = r2_score(z_test, z_pred)
mse = mean_squared_error(z_test, z_pred)

print(f"\nMean Squared Error: {mse}")
print(f"R2-score: {r2}")
