# General imports
from own_code import * 
import numpy as np
from random import random, seed

# Import scikit regression tools
from sklearn.model_selection import train_test_split, cross_validate, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Import plotting tools
import matplotlib.pyplot as plt

# Where to save the figures and data files
PROJECT_ROOT_DIR = "Results"
FIGURE_ID = "Results/FigureFiles"
DATA_ID = "DataFiles/"



# ---MAKE DATA---
k = 20                      # Number of points on each axis
x = np.arange(0, 1, 1/k)    # Numbers on x-axis
y = np.arange(0, 1, 1/k)    # Numbers on y-axis
x, y = np.meshgrid(x,y)     # Create meshgrid of x and y axes
z = FrankeFunction(x,y)     # The z-values

x1 = np.ravel(x)            # Flatten to vector
n = len(x1)                 # Number of observations (n=k*k)
y1 = np.ravel(y)
z1 = np.ravel(z) #+ np.random.random(n)*0.01     # Add noise if wanted

"""
Part a) Linear Regression 
"""
def part_a():

    # ---LINREAR REGRESSION---
    X = CreateDesignMatrix_X(x1, y1, d=5)
    #beta = np.linalg.pinv((X.T.dot(X))).dot(X.T).dot(z1)        
    beta = np.linalg.pinv(X.T @ X) @ X.T @ z1
    p = len(beta)                       # p is the complexity of the model
    z_pred = X @ beta                   # Predicted z values

    # ---MAKE PLOTS---
    # Plot the surface of the Franke function
    make3Dplot(x,y,z, title="Franke Function", name="franke_func.png", show=False)

    # Plot the surface of the predicted values
    make3Dplot(x,y,z_pred.reshape(k,k), title="Linear regression (Franke function)", 
                    name ="lin_reg_franke_func.png", show=False)

    # Plot surface of predicted values and of real z-values in one figure
    make3Dplot_multiple(x,y,(z, z_pred.reshape(k,k)), 
                        title=("Plot of real Franke function", "Plot of z-predictions based on linear model"),
                        name="franke_func_vs_linear_model.png", size=(25,10), show=False)


    # ---MAKE CONFIDENCE INTERVALS---
    # Estimate of the variance of error
    s2 = (1/(n-p-1))*np.sum((z1 - z_pred)**2)                        # Estimate of sigma^2 from Hastia
    s1 = (np.linalg.norm(z) - np.linalg.norm(z_pred))**2/(n - p)    # Sample variance formula from Azzalini
    print(f"s2: {s2}") # Is this one correct to use?
    print(f"s1: {s1}") # Why is this so low? Are they supposed to be the same?


    # Computing the variances.
    variance_beta = s2*np.linalg.inv(X.T.dot(X))    # Covariance matrix                
    beta_var = np.diag(variance_beta)               # Variances of the betas 
    beta_CIs = []                                   # List to contain the confidence intervals

    # Find confidence intervals and print beta values
    print("\nOLS LINEAR REGRESSION (OWN CODE)")
    print("\nCoefficient estimations \t Confidence interval")
    for i in range(p):
        beta_CIs.append([beta[i]-1.96*np.sqrt(beta_var[i]/n), beta[i]+1.96*np.sqrt(beta_var[i]/n)])
        print(f"Beta_{i:2d} = {beta[i]:12.8f} \t\t {beta_CIs[i]}")

    # NB: Someone check that the variances and CI's are correct...

    # ---EVALUATE MSE AND R2---
    print(f"\nMean Squared Error: {MSE(z, z_pred)}")        #MSE using own code
    print(f"R2-score: {R2(z, z_pred)}")                     #R2 using own code 

    print(f"MSE (scikitfunc): {mean_squared_error(z1, z_pred)}")    # For comparison
    print(f"R2 (scikitfunc): {r2_score(z1, z_pred)}")               # For comparison

    # ---COMPARE RESULTS OF OWN CODE WITH SCIKIT---
    linreg = LinearRegression(fit_intercept=False)
    linreg.fit(X, z1)
    p = len(linreg.coef_)
    z_pred_scikit = linreg.predict(X)

    print("\nOLS LINEAR REGRESSION USING SCKIKIT LEARN")
    print("\nCoefficient estimations \t Confidence interval")
    for i in range(p):
        print(f"Beta_{i:2d} = {linreg.coef_[i]:12.8f}")

    # Evaluate MSE and R2
    r2 = r2_score(z1, z_pred_scikit)
    mse = mean_squared_error(z1, z_pred_scikit)
    print(f"\nMean Squared Error: {mse}")
    print(f"R2-score: {r2}")

    # Plot to compare with results from own code
    make3Dplot_multiple(x,y,(z_pred.reshape(k,k), z_pred_scikit.reshape(k,k)), 
                title=("Linear regression with own code","Linear regression with scikit"), 
                name="lin_reg_scikit_franke_func.png", show=True)

    return

#part_a()

"""
Part b) Resampling of data and Cross-validation
"""
# ---SPLIT DATA IN TRAIN/TEST SETS AND PERFORM LINEAR REGRESSION---
test_size = 0.2     # size of test-set (in percentage)
x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=test_size, random_state=1)
X_train = CreateDesignMatrix_X(x_train, y_train, d=5)
X_test = CreateDesignMatrix_X(x_test, y_test, d=5)
z_train = FrankeFunction(x_train, y_train)
z_test = FrankeFunction(x_test, y_test)

beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ z_train
z_pred = X_test @ beta

mse = (1/len(z_test))*((z_test - z_pred).T @ (z_test - z_pred))
#mse = np.mean((z_test - z_pred)**2)
print("\nLINEAR REGRESSION WITH 80% TRAINING DATA AND 20% TEST DATA:")
print(f"mse formula = {mse}")
print(f"mse scikit = {mean_squared_error(z_test, z_pred)}")
print(f"MSE = {MSE(z_test, z_pred)}\nR2 = {R2(z_test, z_pred)}")


# ---CROSS VALIDATION---

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


def k_Cross_Validation(x, y, z, k=5, d=3, seed = None):
    """
    Function that performs k-fold cross-validation with Linear regression
    on data x, y, z=f(x, y) for some function f that we are trying to model.
    d specifies the polynomial degree of the linear model.
        - x, y and z are arrays. k is an integer with default 5. d is an 
        integer with default 3.
    """

    error_test = []
    error_train = []
    bias2 = []
    variance = []
    r2 = []

    n = len(z)              # Number of "observations"
    i = int(n/k)            # Size of test set
    print(f"\nPERFORMING {k}-FOLD CROSS VALIDATION:")
    print(f"Number of observations (n): {n}")
    print(f"Minimum size of test set: {i}")
    print(f"Degree of the polynomial: {d}")

    test_folds = k_folds(n, k=k, seed=seed)
    #model = make_pipeline(PolynomialFeatures(degree=d), LinearRegression(fit_intercept=False))

    z_pred_test_average = np.zeros(i)
    
    for indexes in test_folds:
        m = len(indexes)
        x_test = x[indexes]
        y_test = y[indexes]
        z_test = z[indexes]
        #z_test = FrankeFunction(x_test, y_test)
        x_train = x[np.delete(np.arange(n), indexes)]
        y_train = y[np.delete(np.arange(n), indexes)]
        z_train = z[np.delete(np.arange(n), indexes)]
        #z_train = FrankeFunction(x_train, y_train)

        X_test = CreateDesignMatrix_X(x_test, y_test, d=d)
        X_train = CreateDesignMatrix_X(x_train, y_train, d=d)

        beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ z_train
        z_pred_test = X_test @ beta
        z_pred_train = X_train @ beta
        z_pred_test_average += z_pred_test/k
        #fit = model.fit(X_test, z_test)
        #z_pred_test = fit.predict(X_test)
        #fit = model.fit(X_train, z_train)
        #z_pred_train = fit.predict(X_train)


        error_test.append(sum((z_test - z_pred_test)**2)/len(z_test))
        error_train.append(sum((z_train - z_pred_train)**2)/len(z_train))
        #error_test.append(((z_test - z_pred_test)@(z_test - z_pred_test).T)/len(z_pred))
        #error_train.append(((z_train - z_pred_train)@(z_train - z_pred_train).T)/len(z_pred))
        r2.append(R2(z_test, z_pred_test))
        #variance.append(np.var(z_pred_test))
        #bias2.append((z_test - np.mean(z_pred_test))**2)
    
    #bias = numpy.linalg.norm() 

    test_err = np.mean(error_test)
    train_err = np.mean(error_train)
    bias = np.mean(bias2)
    var = np.mean(variance)
    r2_score = np.mean(r2)

    return test_err, train_err, bias, var, r2_score
    

def part_b():
    test_err, train_err, bias, var, r2_score = k_Cross_Validation(x1, y1, z1)
    print(test_err, train_err)

    # ---MAKE PLOT OF ERROR VS. DEGREE OF POLYNOMIAL---
    maxdegree = 30
    degrees = np.arange(1, maxdegree+1)

    test_err_results = []
    train_err_results = []
    bias_results = []
    variance_results = []
    r2_score_results = []

    for deg in degrees:
        test_err, train_err, bias, var, r2_score = k_Cross_Validation(x1, y1, z1, d=deg, seed = 1001)
        test_err_results.append(test_err)
        train_err_results.append(train_err)
        bias_results.append(bias)
        variance_results.append(var)
        r2_score_results.append(r2_score)

        print('\nPolynomial degree:', deg)
        print('Error:', test_err)
        print('Bias^2:', bias)
        print('Var:', var)
        print(f'{test_err:.4f} >= {bias:.4f} + {var:.4f} = {bias+var:.4f}')

    # Plot
    # Plot test-error and train-error
    plt.figure()
    plt.plot(degrees, test_err_results, 'k', label='Test MSE')
    plt.plot(degrees, train_err_results, 'b', label='Train MSE')
    plt.legend()
    save_fig('train-test-error-plot.png')
    plt.show()

    # Plots bias-variane-mse
    plt.figure()
    plt.plot(degrees, bias_results, label = 'bias^2')
    plt.plot(degrees, variance_results, label = 'variance')
    plt.plot(degrees, test_err_results, label = 'error')
    plt.legend()
    save_fig('bias-variance-plot.png')
    plt.show()



part_b()





"""
Part c)
"""
