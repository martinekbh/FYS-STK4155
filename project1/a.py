'''
Part a) of project 1
'''
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
i = len(x)
x, y = np.meshgrid(x,y)

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

z = FrankeFunction(x, y) #+ 0.05*np.random.randn(i,i) # Calculate z + noise


# Plot the surface.
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title("Funke function")
plt.savefig("funke_func.png")

#plt.show()

# Perform standard least squares regression analysis, using
# polynomials in x and y up to fifth order.
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

#Transform from matrices to vectors
x_1=np.ravel(x)
y_1=np.ravel(y)
n=int(len(x_1))
z_1=np.ravel(z)#+ np.random.random(n) * 1


# finally create the design matrix
X= CreateDesignMatrix_X(x_1,y_1,n=5)
print(X)

# find beta
beta = np.linalg.inv((X.T.dot(X))).dot(X.T).dot(z_1)        # MAYBE USE np.linalg.pinv INSTEAD?
print(beta)
z_pred = X @ beta   # Predicted z values
print("z_pred: ", z_pred)
print("\nZ: ", z)
print("\nx: ", x)
print("\ny: ", y)


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    new_l = np.zeros(n)
    for i in range(0, n):
        new_l[i] = l[i*n:i*n + n]
    return new_l

z_pred = z_pred.reshape(i,i)

# Make plot
fig = plt.figure()
ax = fig.gca(projection='3d')

# Plot the surface.
surf = ax.plot_surface(x, y, z_pred, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title("Linear regression (funke function)")
plt.savefig("lin_reg_funke_func.png")
plt.show()

# SCIKIT:
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



# Find confidence intervals of the betas by computing the variances.



# Evaluate MSE and R^2-score
