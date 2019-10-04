# Import regression tools
from sklearn.model_selection import train_test_split, cross_validate, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
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
def make3Dplot(x, y, zlist, title=None, name=None, size = (16,12), show=True):
    """
    Function for making a plot with multiple 3D subplots. x and z must be np.meshgrid.
    y is a list of np.meshgrids.
    """
    if len(zlist)==2 and size==(16,12):     # Reset size to new default for two subsplots
        size=(25,10)

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

# Convert matrix (meshgrid) to vector
x1 = np.ravel(x)
y1 = np.ravel(y)
n = len(x1)                                 # Number of "observations"
np.random.seed(1)
z1 = np.ravel(z) #+ np.random.random(n)*0.01 # Add noise if wanted
xy = np.concatenate((x1.reshape(len(x1),1), y1.reshape(len(x1), 1)), axis=1)



# Test with normal linear regression:
linreg = make_pipeline(PolynomialFeatures(degree=3), LinearRegression(fit_intercept=False))
linreg.fit(xy, z1)
z_pred = linreg.predict(xy)
mse = mean_squared_error(z1, z_pred)
print(mse)




# CROSS VALIDATION
k=5
kfold = KFold(n_splits=k, shuffle=True, random_state=1)

#Two clumsy lines to get the size of y_pred array right
X_trainz, X_testz, z_trainz, z_testz = train_test_split(x1,z1,test_size=1./k)
array_size_test=len(z_testz)
array_size_train=len(z_trainz)



train_err = []
err = []
bias2 = []
var = []
maxdegree = 10
degrees = np.arange(1, maxdegree+1)


for deg in degrees:
    z_pred = np.empty((array_size_test, k))
    model = make_pipeline(PolynomialFeatures(degree=deg), LinearRegression(fit_intercept=False))
    #X = CreateDesignMatrix_X(x1, y1, n=deg)

    z_train_pred = np.empty((array_size_train, k))

    #errCV = []
    #bias2CV = []
    #varCV = []
    train_errCV = []
    error = 0
    train_error = 0
    bias = 0
    vari = 0

    j=0     # counter
    for train_index, test_index in kfold.split(x1):  # Perform k-fold cross-validation
        x_train = x1[train_index]
        x_test = x1[test_index]
        y_train = y1[train_index]
        y_test = y1[test_index]

        z_train = FrankeFunction(x_train, y_train)
        z_test = FrankeFunction(x_test, y_test)

        # Dette funker ogsaa. Why?
        #z_train = z1[train_index]
        #z_test = z1[test_index]

        X_train = CreateDesignMatrix_X(x_train, y_train)
        X_test = CreateDesignMatrix_X(x_test, y_test)

        fit = model.fit(X_train, z_train)
        #z_train_pred[:,j] = fit.predict(X_train)

        z_pred_train = fit.predict(X_train)
        train_error += np.mean((z_train - z_pred_train)@(z_train - z_pred_train).T)

        z_p = fit.predict(X_test)
        z_pred[:,j] = fit.predict(X_test)
        error += np.mean((z_test - z_p)@(z_test - z_p).T) 
        vari += np.var(z_p)

        #errCV.append(mean_squared_error(z_test, z_pred[:,j]))
        #bias2CV.append((z_test - np.mean(z_pred))**2)
        #varCV.append(np.var(z_pred))

        j += 1  # counter

    #error = np.mean(errCV)
    #bias = np.mean(bias2CV)
    #variance = np.mean(varCV)
    #train_error = np.mean(train_errCV)

    #variance = np.mean( np.var(z_pred, axis=1, keepdims=True) )
    #error = np.mean( np.mean((z_test - z_pred)**2, axis=1, keepdims=True) )
    #bias = np.mean( (z_test - np.mean(z_pred, axis=1, keepdims=True))**2 )

    train_err.append(train_error/k)
    err.append(error/k)
    bias2.append(bias/k)
    var.append(vari/k)
    #err.append(error)
    #bias2.append(bias)
    #var.append(variance)

    print('\nPolynomial degree:', deg)
    print('Error:', error/k)
    print('Bias^2:', bias/k)
    print('Var:', vari/k)
    print(f'{error:.4f} >= {bias/k:.4f} + {vari/k:.4f} = {bias/k+vari/k:.4f}')


# Plot
plt.figure()
plt.plot(degrees, err, 'k', label='MSE')
plt.plot(degrees, bias2, 'b', label='Bias^2')
plt.plot(degrees, var, 'y', label='Var')
summ=np.zeros(len(var))
for i in range(len(err)):
    summ[i]=var[i]+bias2[i]
plt.plot(degrees, summ, 'ro', label='sum')

plt.xlabel('Polynomial degree')
plt.ylabel('MSE CV')
plt.legend()
plt.savefig('bias-variance-decomposition.png')
plt.show()


# Plot test-error and train-error
plt.figure()
plt.plot(degrees, err, 'k', label='Test MSE')
plt.plot(degrees, train_err, 'b', label='Train MSE')
plt.legend()
plt.savefig('train-test-error-plot.png')
plt.show()
