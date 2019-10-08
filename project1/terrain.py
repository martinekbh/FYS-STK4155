from own_code import *
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures


# Where to save the figures and data files
PROJECT_ROOT_DIR = "Results"
FIGURE_ID = "Results/FigureFiles"
DATA_ID = "DataFiles/"

# Load the terrain
#terrain1 = imread(DATA_ID + 'SRTM_data_Norway_1.tif')
terrain1 = imread(DATA_ID + 'SRTM_data_Norway_2.tif')
terrain1_reduced = terrain1[1000:2800,:]
print(terrain1_reduced.shape)


# Show the terrain
plt.figure()
plt.title('Terrain over Norway 1')
plt.imshow(terrain1, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
save_fig('terrain1')
#plt.show()
#plt.close()

# Show the terrain
plt.figure()
plt.title('Terrain over Norway 1')
plt.imshow(terrain1_reduced, cmap='gray', extent=[0, 1801, 1000, 2800])
plt.xlabel('X')
plt.ylabel('Y')
save_fig('terrain1_reduced')
#plt.show()
#plt.close()

# ---MAKE DATA---
y_len, x_len = terrain1_reduced.shape
print(f'Number of x: {x_len}, Number of y: {y_len}')
x = np.arange(x_len)
y = np.arange(y_len)
x, y = np.meshgrid(x, y)
print(x.shape, y.shape)
print(terrain1_reduced.shape)
x1 = np.ravel(x)
y1 = np.ravel(y)
z1 = np.ravel(terrain1_reduced)
n = len(z1)

print(np.amax(z1))
print(np.amin(z1))

'''
# ---LINEAR REGRESSION---
deg = 5
X = CreateDesignMatrix_X(x1, y1, d=deg)
#beta = np.linalg.pinv(X.T @ X) @ X.T @ z1
beta = SVDinv(X.T @ X) @ X.T @ z1 #more stable inversion than above

z_pred = X @ beta

print(np.amax(z_pred))#
print(np.amin(z_pred))#
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

print(f"\nMean Squared Error: {MSE(z1, z_pred)}")    # NB: Someone check that this value is correct...
print(f"R2-score: {R2(z1, z_pred)}")


# make plot
print(z_pred.reshape(y_len, x_len).shape)
plt.figure()
plt.imshow(z_pred.reshape(y_len, x_len), cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Linear regression with degree={deg}')
save_fig('linreg-terrain1')
plt.show()
'''


deg=2
# ---LASSO REGRESSION---
X = CreateDesignMatrix_X(x1, y1, d=deg)

clf = linear_model.Lasso(alpha=0.3)
lasso= clf.fit(X, z1)

z_pred = lasso.predict(X)
r2 = r2_score(z1, z_pred)
mse = mean_squared_error(z1, z_pred)

print(f"\nMean Squared Error: {mse}")
print(f"R2-score: {r2}")

# make plot
print(z_pred.reshape(y_len, x_len).shape)
plt.figure()
plt.imshow(z_pred.reshape(y_len, x_len), cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Lasso regression with degree={deg}')
save_fig('Lassoereg-terrain1')
plt.show()



def linreg_CV():
    # ---CV to find out which degree is best for linear regression---
    maxdegree = 10
    degrees = np.arange(1, maxdegree+1)

    test_err_results = []
    train_err_results = []
    r2_score_results = []

    for deg in degrees:
        print(f"Degree: {deg}")
        test_err, train_err, r2 = k_Cross_Validation(x1, y1, z1, d=deg)
        test_err_results.append(test_err)
        train_err_results.append(train_err)
        r2_score_results.append(r2)

    # Plot test-error and train-error
    plt.figure()
    plt.plot(degrees, test_err_results, 'k', label='Test MSE')
    plt.plot(degrees, train_err_results, 'b--', label='Train MSE')
    plt.legend()
    plt.xlabel('degree of polynomial')
    plt.ylabel('error')
    save_fig('terrain-train-test-error-plot.png')
    plt.title('Training and test error vs. polynomial degree')
    #plt.show()

    #opt_degree = np.where(test_err_results == test_err_results.min())
    opt_degree = 2
    return

def ridge_CV():

    print(f"\nRIDGE REGRESSION ANALYSIS")
    n_lambdas = 5
    lambdas = np.logspace(-4,0, n_lambdas)
    maxdegree = 10
    degrees = np.arange(1, maxdegree+1)
    k = 5

    mse_scores = np.zeros((maxdegree, n_lambdas)) # Matrix to save the mse-scores

    i=0
    for deg in degrees:
        print(deg)
        j=0
        for lmb in lambdas:
            print(lmb)
            test_err, train_err, r2 = k_Cross_Validation(x1, y1, z1, k=k, d=deg, reg_method='Ridge', lmb=lmb)
            mse_scores[i,j] = test_err
            j += 1
        i += 1

    min_MSE = mse_scores.min()
    min_index = np.where(mse_scores == min_MSE)
    deg_ind, lmb_ind = tuple(i.item() for i in min_index)
    opt_lambda = lambdas[lmb_ind]
    opt_degree = degrees[deg_ind]

    print(f"Best MSE: {min_MSE}")
    print(f"with lambda={opt_lambda}, and degree={opt_degree}")

    # Plot
    im = plt.imshow(mse_scores, cmap=plt.cm.RdBu, extent = [lambdas[0], lambdas[-1], degrees[-1], degrees[0]],
                interpolation=None, aspect='auto')
    plt.colorbar(im)
    plt.xlabel('lambda')
    plt.ylabel('degree of polynomial')
    plt.title('MSE colormap (Ridge)')
    save_fig('terrain-ridge-degree-lambda-colormap')
    plt.show()

    """

    # ---RIDGE REGRESSION with optimal parameters---
    X = CreateDesignMatrix_X(x1, y1, d=opt_degree)
    XTX = X.T @ X
    dim = len(XTX)
    W = np.linalg.pinv(XTX + opt_lambda*np.identity(dim))
    beta = W @ X.T @ z1
    p = len(beta)                       # p is the complexity of the model
    z_pred = X @ beta                   # Predicted z values


    # ---MAKE CONFIDENCE INTERVALS FOR THE BETAS---
    # Estimate of the variance of error
    s2 = (1/(n-p-1))*np.sum((z1 - z_pred)**2)                        # Estimate of sigma^2 from Hastia
    print(f"s2: {s2}") # Is this one correct to use?


    # Computing the variances.
    variance_beta = s2*(W @ XTX @ W.T)              # Covariance matrix
    beta_var = np.diag(variance_beta)               # Variances of the betas
    beta_CIs = []                                   # List to contain the confidence intervals

    # Find confidence intervals and print beta values
    print("\nRIDGE REGRESSION ON WHOLE DATASET WITH OPTIMAL PARAMETERS")
    print(f"Lambda = {opt_lambda:.6f}, polynomial degree= = {opt_degree}")
    print("\nCoefficient estimations \t Confidence interval")
    for i in range(p):
        beta_CIs.append([beta[i]-1.96*np.sqrt(beta_var[i]/n), beta[i]+1.96*np.sqrt(beta_var[i]/n)])
        print(f"Beta_{i:2d} = {beta[i]:12.8f} \t\t {beta_CIs[i]}")
    """

    return

#linreg_CV()
#ridge_CV()
