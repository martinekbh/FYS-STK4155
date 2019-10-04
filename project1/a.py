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
def make3Dplot(x, y , z, title=None, name=None, size=(16,12),  show=True):
    """
    Function for making 3D plot. x, y, and z must be np.meshgrid
    """
    fig = plt.figure(figsize=size)
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



def make3Dplot_multiple(x, y, zlist, title=None, name=None, size=(25,10), show=True):
    """
    Function for making a plot with multiple 3D subplots. x and z must be np.meshgrid.
    y is a list of np.meshgrids.
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
        plt.savefig(name)
    if show:
        plt.show()
    else:
        plt.close()

# Plot the surface of the Franke function
make3Dplot(x,y,z, title="Franke Function", name="franke_func.png", show=False)

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
x1 = np.ravel(x)
n = len(x1)        # Number of observations (n=k*k, see beginning of program)
y1 = np.ravel(y)
z1 = np.ravel(z) + np.random.random(n)*0.01 # Add noise if wanted

# Create the design matrix
X = CreateDesignMatrix_X(x1,y1,n=5)   # n is the degree of the model polynomial 

# Find beta and predict z-values
beta = np.linalg.inv((X.T.dot(X))).dot(X.T).dot(z1)        # MAYBE USE np.linalg.pinv INSTEAD?
p = len(beta)                       # p is the complesity of the model
z_pred = X @ beta                   # Predicted z values
z_pred = z_pred.reshape(k,k)        # Make z into appropriate matrix so we can plot it

# Plot the surface of the predicted values
make3Dplot(x,y,z_pred, title="Linear regression (Franke function)", 
                name ="lin_reg_franke_func.png", show=False)

# Plot surface of predicted values and of real z-values in one figure
make3Dplot_multiple(x,y,(z, z_pred.reshape(k,k)), 
                    title=("Plot of real Franke function", "Plot of z-predictions based on linear model"),
                    name="franke_func_vs_linear_model.png", size=(25,10), show=False)

# Find confidence intervals of the betas by computing the variances.
s2 = (1/(n-p-1))*np.sum((z - z_pred)**2)                          # Estimate of sigma^2 from Hastia
s1 = (np.linalg.norm(z) - np.linalg.norm(z_pred))**2/(n - p)    # Sample variance formula from Azzalini
print(f"s2: {s2}") # Is this one correct to use?
print(f"s1: {s1}") # Why is this so low? Are they supposed to be the same?


variance_beta = s2*np.linalg.inv(X.T.dot(X))    # Covariance matrix                
beta_var = np.diag(variance_beta)               # Variances of the betas 
#print(beta_var)
beta_CIs = []                                   # List to contain the confidence intervals

# Find confidence intervals and print beta values
print("\nOLS LINEAR REGRESSION (OWN CODE)")
print("\nCoefficient estimations \t Confidence interval")
for i in range(p):
    beta_CIs.append([beta[i]-1.96*np.sqrt(beta_var[i]/n), beta[i]+1.96*np.sqrt(beta_var[i]/n)])
    print(f"Beta_{i:2d} = {beta[i]:12.8f} \t\t {beta_CIs[i]}")

# NB: Someone check that the variances and CI's are correct...



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

print(f"\nMean Squared Error: {MSE(z, z_pred)}")    # NB: Someone check that this value is correct...
print(f"R2-score: {R2(z, z_pred)}")                 # NB: Someone check that this value is correct...


# SCIKIT:
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# SKIKIT OLS Linear Regression (for comparisom with own code)
linreg = LinearRegression(fit_intercept=False)
linreg.fit(X, z1)
p = len(linreg.coef_)

print("\nOLS LINEAR REGRESSION USING SCKIKIT LEARN")
print("\nCoefficient estimations \t Confidence interval")
for i in range(p):
    print(f"Beta_{i:2d} = {linreg.coef_[i]:12.8f}")


z_pred_scikit = linreg.predict(X)
r2 = r2_score(z1, z_pred_scikit)
mse = mean_squared_error(z1, z_pred_scikit)
print(f"\nMean Squared Error: {mse}")
print(f"R2-score: {r2}")

k = int(np.sqrt(len(z_pred_scikit)))
z_pred_scikit = z_pred_scikit.reshape(k,k)
make3Dplot_multiple(x,y,(z_pred, z_pred_scikit), title=("Linear regression with own code","Linear regression with scikit"), 
            name="lin_reg_scikit_franke_func.png", show=False)



# TEST OTHER METHOD OF ORGANIZING THE DATA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

xy = np.concatenate((x1.reshape(len(x1),1), y1.reshape(len(x1), 1)), axis=1)
linreg = make_pipeline(PolynomialFeatures(degree=3), LinearRegression(fit_intercept=False))
linreg.fit(xy, z1)
z_pred = linreg.predict(xy)
mse = mean_squared_error(z1, z_pred)
r2 = r2_score(z1, z_pred)
print("\nTest:")
print(f"MSE : {mse}")
print(f"R2: {r2}")

