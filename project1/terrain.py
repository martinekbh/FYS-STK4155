# General imports
from own_code import *
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time

# Scikit imports
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures


# Where to save the figures and data files
PROJECT_ROOT_DIR = "Results"
FIGURE_ID = "Results/FigureFiles"
DATA_ID = "DataFiles/"

# Load the terrain data
#terrain1 = imread(DATA_ID + 'SRTM_data_Norway_1.tif')
terrain = imread(DATA_ID + 'SRTM_data_Norway_2.tif')
terrain_reduced = terrain[1500:2700,:1100]


# Show the terrain
plt.figure()
plt.title('Terrain over Norway')
plt.imshow(terrain, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
save_fig('terrain')
plt.show()

# Show the terrain reduced
plt.figure()
plt.title('Terrain over Norway')
plt.imshow(terrain_reduced, cmap='gray', extent=[0,1100,2700,1500])
plt.xlabel('X')
plt.ylabel('Y')
save_fig('terrain_reduced')
plt.show()


# ---MAKE DATA---
y_len, x_len = terrain_reduced.shape
print(f'Number of x: {x_len}, Number of y: {y_len}')

x = np.arange(x_len)            # x-axis
xmax = x/(np.amax(x))           # Normalize data
y = np.arange(y_len)            # y-axis
ymax = y/(np.amax(y))           # Normalize data
x, y = np.meshgrid(xmax, ymax)  # Make into grid
x1 = np.ravel(x)                # Flatten to vector
y1 = np.ravel(y)                # ---''---
z1 = np.ravel(terrain_reduced)  # ---''---
n = len(z1)                     # n is the number of datapoint/observations


deg = 5

"""
# ---LINEAR REGRESSION---
t2 = time.time()
X = CreateDesignMatrix_X(x1, y1, d=deg)
#beta = np.linalg.pinv(X.T @ X) @ X.T @ z1
beta = SVDinv(X.T @ X) @ X.T @ z1 #more stable inversion than above
z_pred = X @ beta

t3=time.time()
print("Runtime in seconds: {}".format(t3-t2))

# Plot
plt.figure()
plt.imshow(z_pred.reshape(y_len, x_len), cmap='gray', extent=[0,1100,2700,1500])
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Linear regression with degree={deg}')
save_fig('terrain-linreg')
plt.show()

print(f'Linear regression with degree {deg}')
print(f"\nMean Squared Error: {MSE(z1,z_pred)}")
print(f"R2-score: {R2(z1,z_pred)}")

# ---MAKE CONFIDENCE INTERVALS---
p = len(beta)
s2 = (1/(n-p-1))*np.sum((z1 - z_pred)**2)       # Estimate of sigma^2 from Hastia

# Computing the variances.
variance_beta = s2*np.linalg.inv(X.T.dot(X))    # Covariance matrix                
beta_var = np.diag(variance_beta)               # Variances of the betas 
beta_CIs = np.zeros((p,2))                      # Array to contain the confidence intervals

# Find confidence intervals and print beta values
print("\nCoefficient estimations \t Confidence interval")
for i in range(p):
    beta_CIs[i,0] = beta[i]-1.96*np.sqrt(beta_var[i]/n)
    beta_CIs[i,1] = beta[i]+1.96*np.sqrt(beta_var[i]/n)
    print(f"Beta_{i:2d} = {beta[i]:12.8f} \t\t {beta_CIs[i,:]}")

"""

# ---RIDGE REGRESSION---

"""

X = CreateDesignMatrix_X(x1, y1, d=deg)
XTX = X.T @ X
dim = len(XTX)
W = np.linalg.pinv(XTX + 0.01*np.identity(dim)) #chose a lambda
beta = W @ X.T @ z1
p = len(beta)                       # p is the complexity of the model
z_pred = X @ beta                   # Predicted z values

print(z_pred.reshape(y_len, x_len).shape)
plt.figure()
plt.imshow(z_pred.reshape(y_len, x_len), cmap='gray', extent=[0,1100,2700,1500])
plt.xlabel('X')
plt.ylabel('Y')
#plt.title(f'Ridge regression with degree={deg}')
save_fig(f'terrain-ridge-deg{deg}')
plt.show()

print('linear regression data')
print(f"\nMean Squared Error: {MSE(z1,z_pred)}")
print(f"R2-score: {R2(z1,z_pred)}")

"""


# ---LASSO REGRESSION---  #very slow, dont try higher than degree 8

'''
X = CreateDesignMatrix_X(x1, y1, d=deg)
lmb = 0.2
clf = linear_model.Lasso(alpha=lmb)
lasso= clf.fit(X, z1)

z_pred = lasso.predict(X)
r2 = r2_score(z1, z_pred)
mse = mean_squared_error(z1, z_pred)

print(f"\nMean Squared Error: {mse}")
print(f"R2-score: {r2}")
t1=time.time()
print("Runtime in seconds: {}".format(t1-t0))

# make plot
plt.figure()
plt.imshow(z_pred.reshape(y_len, x_len), cmap='gray', extent=[0,1100,2700,1500])
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Lasso regression with degree={deg}')
save_fig(f'terrain-lasso-lmb{lmb:.4f}-deg{deg}')
plt.show()
'''

def linreg_CV():
    print('\nOLS LINEAR REGRESSION ANALYSIS')
    # ---CV to find out which degree is best for linear regression---
    maxdegree = 40
    degrees = np.arange(1, maxdegree+1)
    print(f'Initializing CV with OLS for degrees d=1, ..., {maxdegree} to plot MSE:')

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
    save_fig('terrain-train-test-error-plot')
    #plt.title('Training and test error vs. polynomial degree')
    #plt.show()

    # Record the optimal degree d
    min_index = np.where(test_err_results == min(test_err_results))
    deg_ind = tuple(i.item() for i in min_index)
    opt_degree = degrees[deg_ind]       # Degree which gives lowest MSE

    print('---done---')
    print(f"Best MSE {min(test_err_results)} at degree={opt_degree}")

    # ---LINEAR REGRESSION with optimal degree---
    X = CreateDesignMatrix_X(x1, y1, d=opt_degree)
    beta = SVDinv(X.T @ X) @ X.T @ z1
    z_pred = X @ beta

    # PLOT
    plt.figure()
    plt.imshow(z_pred.reshape(y_len, x_len), cmap='gray', extent=[0,1100,2700,1500])
    plt.xlabel('X')
    plt.ylabel('Y')
    save_fig('terrain-linreg-optdegree')
    #plt.show()

    print(f'\nLinear regression with degree {opt_degree} on whole dataset:')
    print(f"Mean Squared Error: {MSE(z1,z_pred)}")
    print(f"R2-score: {R2(z1,z_pred)}")


    # ---MAKE CONFIDENCE INTERVALS---
    s2 = (1/(n-p-1))*np.sum((z1 - z_pred)**2)       # Estimate of sigma^2 from Hastia

    # Computing the variances.
    variance_beta = s2*np.linalg.inv(X.T.dot(X))    # Covariance matrix                
    beta_var = np.diag(variance_beta)               # Variances of the betas 
    beta_CIs = np.zeros((p,2))                      # Array to contain the confidence intervals

    # Find confidence intervals and print beta values
    print("\nCoefficient estimations \t Confidence interval")
    for i in range(p):
        beta_CIs[i,0] = beta[i]-1.96*np.sqrt(beta_var[i]/n)
        beta_CIs[i,1] = beta[i]+1.96*np.sqrt(beta_var[i]/n)
        print(f"Beta_{i:2d} = {beta[i]:12.8f} \t\t {beta_CIs[i,:]}")

    # ---Make plot of the betas and the CIs---
    beta_lower = beta_CIs[:,0]
    beta_upper = beta_CIs[:,1]
    indexes = np.arange(p)
    plt.plot(indexes, beta, 'ro', label='betas')
    plt.plot(indexes, beta_upper, 'b+')
    plt.plot(indexes, beta_lower, 'b+')
    plt.xlabel('beta index')
    save_fig('terrain-linreg-beta-values-plot')

    print('\a')     # Make alert sound to indicate code is finally done
    return

def ridge_CV():

    print(f"\nRIDGE REGRESSION ANALYSIS")
    n_lambdas = 7
    lambdas = np.logspace(-6,0, n_lambdas)
    maxdegree = 20
    degrees = np.arange(1, maxdegree+1)
    k = 5
    print(f'Initializing CV with Ridge for {n_lambdas} lambdas in [10^(-4),1] and degrees from 1 to {maxdegree}')

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

    # Record optimal degree and lambda
    min_MSE = mse_scores.min()
    min_index = np.where(mse_scores == min_MSE)
    deg_ind, lmb_ind = tuple(i.item() for i in min_index)
    opt_lambda = lambdas[lmb_ind]   # lambda which gives lowest MSE
    opt_degree = degrees[deg_ind]   # degree which gives lowest MSE

    print('---done---')
    print(f"Best MSE: {min_MSE}")
    print(f"with lambda={opt_lambda}, and degree={opt_degree}")

    # Plot colormap of MSE
    im = plt.imshow(mse_scores, cmap=plt.cm.RdBu, extent = [lambdas[0], lambdas[-1], degrees[-1], degrees[0]],
                interpolation=None, aspect='auto')
    plt.colorbar(im)
    plt.xlabel('lambda')
    plt.ylabel('degree of polynomial')
    #plt.title('MSE colormap (Ridge)')
    save_fig('terrain-ridge-degree-lambda-colormap')
    #plt.show()


    # ---RIDGE REGRESSION with optimal parameters---
    X = CreateDesignMatrix_X(x1, y1, d=opt_degree)
    XTX = X.T @ X
    dim = len(XTX)
    W = np.linalg.pinv(XTX + opt_lambda*np.identity(dim))
    beta = W @ X.T @ z1
    p = len(beta)               # p is the complexity of the model
    z_pred = X @ beta           # Predicted z values

    # PLOT
    plt.figure()
    plt.imshow(z_pred.reshape(y_len, x_len), cmap='gray', extent=[0,1100,2700,1500])
    plt.xlabel('X')
    plt.ylabel('Y')
    save_fig('terrain-ridge-optparams')
    #plt.show()

    # ---MAKE CONFIDENCE INTERVALS FOR THE BETAS---
    s2 = (1/(n-p-1))*np.sum((z1 - z_pred)**2)       # Estimate of sigma^2 from Hastia

    # Computing the variances.
    variance_beta = s2*(W @ XTX @ W.T)              # Covariance matrix
    beta_var = np.diag(variance_beta)               # Variances of the betas
    #beta_CIs = []                                   # List to contain the confidence intervals
    beta_CIs = np.zeros((p,2))                      # Array to contain the confidence intervals

    # Find confidence intervals and print beta values
    print("\nRIDGE REGRESSION ON WHOLE DATASET WITH OPTIMAL PARAMETERS")
    print(f"Lambda = {opt_lambda:.6f}, polynomial degree= = {opt_degree}")

    print("\nCoefficient estimations \t Confidence interval")
    #for i in range(p):
    #    beta_CIs.append([beta[i]-1.96*np.sqrt(beta_var[i]/n), beta[i]+1.96*np.sqrt(beta_var[i]/n)])
    #    print(f"Beta_{i:2d} = {beta[i]:12.8f} \t\t {beta_CIs[i]}")
    for i in range(p):
        beta_CIs[i,0] = beta[i]-1.96*np.sqrt(beta_var[i]/n)
        beta_CIs[i,1] = beta[i]+1.96*np.sqrt(beta_var[i]/n)
        print(f"Beta_{i:2d} = {beta[i]:12.8f} \t\t {beta_CIs[i,:]}")

    # ---Make plot of the betas and the CIs---
    beta_lower = beta_CIs[:,0]
    beta_upper = beta_CIs[:,1]
    indexes = np.arange(p)
    plt.plot(indexes, beta, 'ro', label='betas')
    plt.plot(indexes, beta_upper, 'b+')
    plt.plot(indexes, beta_lower, 'b+')
    plt.xlabel('beta index')
    save_fig('terrain-ridge-beta-values-plot')

    print('\a') # Make alert sound to indicate code is finally done
    return

linreg_CV()
#ridge_CV()
