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


# Convert matrix (meshgrid) to vector
x1 = np.ravel(x)
y1 = np.ravel(y)
n = len(x1)                                 # Number of "observations"
np.random.seed(1)
z1 = np.ravel(z) + np.random.random(n)*0.01 # Add noise if wanted


x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.2, random_state=1)
X_train = CreateDesignMatrix_X(x_train, y_train, n=2)
X_test = CreateDesignMatrix_X(x_test, y_test, n=2)
z_train = FrankeFunction(x_train, y_train)
z_test = FrankeFunction(x_test, y_test)

beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ z_train
print(beta)


X = CreateDesignMatrix_X(x1, y1, n=2)

# Split into train and test data
print("LINEAR REGRESSION WITH 80% TRAINING DATA AND 20% TEST DATA:")
X_train, X_test, z_train, z_test = train_test_split(X,z1, test_size=0.2, random_state=1)
P = np.linalg.pinv(X_train.T.dot(X_train)).dot(X_train.T)
beta = P.dot(z_train)
print(f"beta = {beta}")
z_pred = X_test @ beta              # Test model on test data
print(f"MSE = {MSE(z_test, z_pred)}\nR2 = {R2(z_test, z_pred)}")



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
    extra_data_points = len(z) % k      # remainder if the number of points is not divisible by k
    print(f"Ekstra data points: {extra_data_points}")

    for i in range(k-1):
        if extra_data_points > 0:       # If there are extra points, add one to group
            X_train, X_test, z_train, z_test = train_test_split(X_pool, z_pool, test_size=test_size+1)
            extra_data_points -= 1      # decrement
        else:                           # There are no more extra points
            X_train, X_test, z_train, z_test = train_test_split(X_pool, z_pool, test_size=test_size)

        # Add the test sets to the lists
        X_test_sets.append(X_test)
        z_test_sets.append(z_test)

        # Update pool of data points to split later
        X_pool = X_train            
        z_pool = z_train

    # Append the last group
    X_test_sets.append(X_pool) 
    z_test_sets.append(z_pool)

    return X_test_sets, z_test_sets


def k_Cross_Validation(X, z, k=5):
    """
    Function that performs k-fold cross validation.
    X is the design matrix, z is the result variable.
    """
    print(f"\nPERFORMING {k}-FOLD CROSS VALIDATION:")
    n = len(z)                                          # Number of "observations"
    i = int(n/k)                                        # Size of test set
    print(f"Number of observations (n): {n}")
    print(f"Minimum size of test set: {i}")
    X_groups, z_groups = split_data(X, z, k)            # Split data into k groups


    """
    # Check that the groups are correct
    for i in range(len(z_groups)):
        print(f"\nGruppe {i}:")
        print(f"[1, x, y, x^2, xy, y^2] \t\t z-value \t\t FrankeFunction(x,y)")
        for j in range(len(z_groups[i])):
            f = FrankeFunction(X_groups[i][j][1], X_groups[i][j][2])        #test z values
            print(f"{X_groups[i][j]}\t{z_groups[i][j]}\t{f}")
        print(f"Size: {len(z_groups[i])} observations.")
    """

    mse = []    # List to store the MSEs
    r2 = []     # List to store the R2-scores
    bias2 = []
    variance = []

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

        P = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T)
        beta = P.dot(z_train)
        print(f"beta = {beta}")

        z_pred = X_test @ beta                      # Test model on test data

        mse.append(MSE(z_test, z_pred))             # MSE
        r2.append(R2(z_test, z_pred))               # R2-score
        bias2.append((z_test - np.mean(z_pred))**2) # squared bias
        variance.append(np.var(z_pred))             # variance
        print(f"MSE = {mse[i]}\nR2 = {r2[i]}")

    return mse, r2, bias2, variance

mse, r2, bias2, variance = k_Cross_Validation(X, z1, k=5)    

estimated_mse_CV = np.mean(mse)
estimated_bias2_CV = np.mean(bias2)
estimated_variance_CV = np.mean(variance)
print(f"\nSUMMARY OF THE k-FOLD CROSS VALIDATION:")
print(f"Estimated MSE of the model: {estimated_mse_CV}")
print(f"Estimated R2-score of the model: {np.mean(r2)}")
print(f"Estimated squared bias of the model: {estimated_bias2_CV}")
print(f"Estimated variance of the model: {estimated_variance_CV}")
print(f"bias^2 + var <= MSE: {estimated_bias2_CV:.4f} + {estimated_variance_CV:.4f} = {estimated_bias2_CV + estimated_variance_CV:.4f} <= {estimated_mse_CV:.4f}")          # This does not add up...


'''
#poly = PolynomialFeatures(degree = 5)

# CROSS VALIDATION USING SCIKIT?
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_validate, cross_val_score

linreg = LinearRegression(fit_intercept=False)
#max_poly_degree = 30


cv_results = cross_validate(linreg, X, z1, cv=5, scoring=('r2', 'neg_mean_squared_error'), return_train_score=True)
#cv_test_scores = cv_results['test_score']
#cv_train_scores = cv_results['train_score']
#print(f"Scikit score: {np.mean(cv_test_scores)}")   # ???
print(cv_results, '\n')
print(np.mean(mse), '\n')
print(np.mean(cv_results['test_neg_mean_squared_error']))
#print(cv_test_scores)

'''





"""
Part c)
"""
maxdegree = 10
error = np.zeros(maxdegree)
bias = np.zeros(maxdegree)
variance = np.zeros(maxdegree)
polydegree = np.zeros(maxdegree)

for degree in range(maxdegree):
    X = CreateDesignMatrix_X(x1, y1, n=degree)

    # Split into train and test data
    X_train, X_test, z_train, z_test = train_test_split(X,z1, test_size=0.2, random_state=1)

    #P = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T)
    #beta = P.dot(z_train)
    #z_pred = X_test @ beta              # Test model on test data

    linreg = LinearRegression(fit_intercept=False)
    linreg.fit(X_train, z_train)
    z_pred = linreg.predict(X_test)

    polydegree[degree] = degree
    error[degree] = MSE(z_test, z_pred)
    bias[degree] = np.mean((z_test - np.mean(z_pred))**2)
    variance[degree] = np.var(z_pred)

    '''
    print('Polynomial degree:', degree)
    print('Error:', error[degree])
    print('Bias^2:', bias[degree])
    print('Var:', variance[degree])
    print('{} >= {} + {} = {}'.format(error[degree], bias[degree], variance[degree], bias[degree]+variance[degree]))
    '''


plt.plot(polydegree, error, label='Error')
plt.plot(polydegree, bias, label='Bias^2')
plt.plot(polydegree, variance, label='Variance')
plt.legend()
plt.show()
