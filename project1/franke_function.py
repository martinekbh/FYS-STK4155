# General imports
from own_code import * 
import numpy as np
from random import random, seed

# Import scikit regression tools
from sklearn.model_selection import train_test_split, cross_validate, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
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
#xmax = x/(np.amax(x))       # To normalize the x-axis
#ymax = y/(np.amax(y))       # To normalize the y-axis
x, y = np.meshgrid(x,y)       # Create meshgrid of x and y axes
z = FrankeFunction(x,y)             # The z-values

x1 = np.ravel(x)            # Flatten to vector
y1 = np.ravel(y)            # Flatten to vector
n = len(x1)                 # Number of observations (n=k*k)
np.random.seed(1001)        # Set seed for reproducability
z1 = np.ravel(z) + np.random.normal(0, .25, n)    # Add noise if wanted





"""
Part a) Linear Regression 
"""
def part_a():

    # ---LINREAR REGRESSION---
    X = CreateDesignMatrix_X(x1, y1, d=5)
    beta = np.linalg.pinv(X.T @ X) @ X.T @ z1
    p = len(beta)                       # p is the complexity of the model
    z_pred = X @ beta                   # Predicted z values

    # ---MAKE PLOTS---
    # Plot the surface of the Franke function
    make3Dplot(x,y,z, name="franke_func", show=False)

    # Plot the surface of the predicted values
    make3Dplot(x,y,z_pred.reshape(k,k), 
                    name ="lin_reg_franke_func", show=False)

    # Plot surface of predicted values and of real z-values in one figure
    make3Dplot_multiple(x,y,(z, z_pred.reshape(k,k)), 
                    name="franke_func_vs_linear_model", size=(25,10), show=False)


    # ---MAKE CONFIDENCE INTERVALS---
    # Estimate of the variance of error
    s2 = (1/(n-p-1))*np.sum((z1 - z_pred)**2)                        # Estimate of sigma^2 from Hastia
    #s1 = (np.linalg.norm(z) - np.linalg.norm(z_pred))**2/(n - p)    # Sample variance formula from Azzalini
    print(f"s2: {s2}") # Is this one correct to use?
    #print(f"s1: {s1}") # Why is this so low? Are they supposed to be the same?


    # Computing the variances.
    variance_beta = s2*np.linalg.inv(X.T.dot(X))    # Covariance matrix                
    beta_var = np.diag(variance_beta)               # Variances of the betas 
    beta_CIs = np.zeros((p,2))                      # Array to contain the confidence intervals

    # Find confidence intervals and print beta values
    print("\nOLS LINEAR REGRESSION (OWN CODE)")
    print("\nCoefficient estimations \t Confidence interval")
    for i in range(p):
        beta_CIs[i,0] = beta[i]-1.96*np.sqrt(beta_var[i]/n)
        beta_CIs[i,1] = beta[i]+1.96*np.sqrt(beta_var[i]/n)
        print(f"Beta_{i:2d} = {beta[i]:12.8f} \t\t {beta_CIs[i,:]}")

    # NB: Someone check that the variances and CI's are correct...

    # ---Make plot of the betas and the CIs---
    beta_lower = beta_CIs[:,0]
    beta_upper = beta_CIs[:,1]
    print(beta_lower)
    indexes = np.arange(p)
    plt.plot(indexes, beta, 'ro', label='betas')
    plt.plot(indexes, beta_upper, 'b+')
    plt.plot(indexes, beta_lower, 'b+')
    plt.xlabel('beta index')
    save_fig('linreg-beta-values-plot')
    plt.show()

    # ---EVALUATE MSE AND R2---
    print(f"\nMean Squared Error: {MSE(z1, z_pred)}")        #MSE using own code
    print(f"R2-score: {R2(z1, z_pred)}")                     #R2 using own code 

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
                name="lin_reg_scikit_franke_func", show=False)

    return


"""
Part b) Resampling of data and Cross-validation
"""


# ---CROSS VALIDATION---
def part_b():

    np.random.seed(1)
    # ---SPLIT DATA IN TRAIN/TEST SETS AND PERFORM LINEAR REGRESSION---
    test_size = 0.2     # size of test-set (in percentage)
    #x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=test_size, random_state=1)
    test_inds, train_inds = test_train_index(n, test_size=test_size)

    x_train = x1[train_inds]
    x_test = x1[test_inds]
    y_train = y1[train_inds]
    y_test = y1[test_inds]
    z_train = z1[train_inds]
    z_test = z1[test_inds]

    X_train = CreateDesignMatrix_X(x_train, y_train, d=5)
    X_test = CreateDesignMatrix_X(x_test, y_test, d=5)

    beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ z_train
    z_pred = X_test @ beta

    # Plot
    make3Dplot(x_test, y_test, z_pred, title="Linear regression with 80% training data and 20% test data",
                    name="linreg-traintestsplit-20percent-plot", show=False)


    mse = (1/len(z_test))*((z_test - z_pred).T @ (z_test - z_pred))
    #mse = np.mean((z_test - z_pred)**2)
    print("\nLINEAR REGRESSION WITH 80% TRAINING DATA AND 20% TEST DATA:")
    print(f"mse formula = {mse}")
    print(f"mse scikit = {mean_squared_error(z_test, z_pred)}")
    print(f"MSE = {MSE(z_test, z_pred)}\nR2 = {R2(z_test, z_pred)}")

    # ---USE CROSS VALIDATION TO IMPROVE ACCURACY OF RESULTS (MSE)---
    de = 5
    mse_test, mse_train, r2sco = k_Cross_Validation(x1, y1, z1, d=de)
    print(f"\nLINEAR REGRESSION (with 5-fold CV) USING DEGREE {de}:")
    print(f"MSE = {mse_test}\nR2 = {r2sco}")

    # ---MAKE PLOT OF ERROR VS. DEGREE OF POLYNOMIAL---
    maxdegree =16
    degrees = np.arange(1, maxdegree+1)

    test_err_results = []
    train_err_results = []
    r2_score_results = []

    print(f"\nTesting which degree from 1 to {maxdegree} gives lowest MSE using 5-fold CV...")
    for deg in degrees:
        test_err, train_err, r2 = k_Cross_Validation(x1, y1, z1, d=deg)
        test_err_results.append(test_err)
        train_err_results.append(train_err)
        r2_score_results.append(r2)


    # Plot
    # Plot test-error and train-error
    plt.figure()
    plt.plot(degrees, test_err_results, 'k', label='Test MSE')
    plt.plot(degrees, train_err_results, 'b', label='Train MSE')
    plt.legend()
    plt.xlabel('degree of polynomial')
    plt.ylabel('error')
    save_fig('linreg-train-test-error-plot')
    #plt.title('training and test error vs. polynomial degree')
    plt.show()

    # ---DO LINEAR REGRESSION FOR OPTIMAL DEGREE d---
    min_index = np.where(test_err_results == min(test_err_results))
    deg_ind  = tuple(i.item() for i in min_index)
    opt_degree = degrees[deg_ind]
    print("Optimal degree: ", opt_degree)

    test_err, train_err, r2 = k_Cross_Validation(x1, y1, z1, d=opt_degree)
    print(f"\nLINEAR REGRESSION (with 5-fold CV) USING OPTIMAL DEGREE {opt_degree}:")
    print(f"MSE = {test_err}\nR2 = {r2}")



    return




"""
Part c) BIAS-VARIANCE PLOT NOT WORKING YET
"""

def part_c():
    maxdegree = 20
    degrees = np.arange(1, maxdegree+1)

    """
    maxdegree = 30
    degrees = np.arange(1, maxdegree+1)

    mse = []
    bias2 = []
    variance = []

    for deg in degrees:
        test_size = 0.2     # size of test-set (in percentage)
        test_indexes, train_indexes = test_train_index(n, test_size=test_size, seed=1)
        x_train = x1[train_indexes]
        x_test = x1[test_indexes]
        y_train = y1[train_indexes]
        y_test = y1[test_indexes]
        z_train = z1[train_indexes]
        z_test = z1[test_indexes]
        #z_train = FrankeFunction(x_train, y_train)
        z_true = FrankeFunction(x_test, y_test)

        X_train = CreateDesignMatrix_X(x_train, y_train, d=deg)
        X_test = CreateDesignMatrix_X(x_test, y_test, d=deg)

        beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ z_train
        z_pred = X_test @ beta


        #bi = (1/len(z_test))*sum((z_test - np.mean(z_pred))**2)
        bi = np.mean((z_true - np.mean(z_pred))**2)
        var = np.mean((z_pred - np.mean(z_pred))**2)
        mse.append(mean_squared_error(z_test, z_pred))
        bias2.append(bi)
        variance.append(var)

    plt.plot(degrees, mse, label='MSE')
    plt.plot(degrees, bias2, label='bias^2')
    plt.plot(degrees, variance, label='Var')
    plt.legend()
    plt.show()
    """

    return



"""
Part d) Ridge
"""

def part_d():

    print(f"\nRIDGE REGRESSION ANALYSIS")
    n_lambdas = 80
    lambdas = np.logspace(-4,0, n_lambdas)
    maxdegree = 15
    degrees = np.arange(1, maxdegree+1)
    k = 5

    mse_scores = np.zeros((maxdegree, n_lambdas)) # Matrix to save the mse-scores

    i=0
    for deg in degrees:
        j=0
        for lmb in lambdas:
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
    plt.show()


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
    #s1 = (np.linalg.norm(z) - np.linalg.norm(z_pred))**2/(n - p)    # Sample variance formula from Azzalini
    print(f"s2: {s2}") # Is this one correct to use?
    #print(f"s1: {s1}") # Why is this so low? Are they supposed to be the same?


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

    return



"""
Part e) Lasso
"""

def part_e():

    print(f"\nLASSO ANALYSIS")
    n_lambdas = 80
    lambdas = np.logspace(-4,0, n_lambdas)
    maxdegree = 15
    degrees = np.arange(1, maxdegree+1)
    k = 5

    mse_scores = np.zeros((maxdegree, n_lambdas)) # Matrix to save the mse-scores

    i=0
    for deg in degrees:
        j=0
        for lmb in lambdas:
            test_err, train_err, r2 = k_Cross_Validation(x1, y1, z1, k=k, d=deg, reg_method='Lasso', lmb=lmb)
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
    plt.title('MSE colormap (Lasso)')
    plt.show()

    # ---LASSO REGRESSION with optimal parameters---
    X = CreateDesignMatrix_X(x1, y1, d=opt_degree)
    lasso = Lasso(alpha=opt_lambda, fit_intercept=False, tol=0.001, max_iter=10e6)
    lasso.fit(X, z1)
    z_pred = lasso.predict(X)

    print(f"\nLASSO REGRESSION ON WHOLE DATASET WITH OPTIMAL PARAMETERS")
    print(f"MSE: {MSE(z1, z_pred)}")
    print(f"R2: {R2(z1, z_pred)}")



    """
    X = CreateDesignMatrix_X(x1, y1, d=5)
    X_train, X_test, z_train, z_test = train_test_split(X,z1, test_size=0.2, random_state=1)

    #Scikits lasso with lambda/alpha value of our choice
    clf = Lasso(alpha=0.2, fit_intercept=False)
    lasso = clf.fit(X_train, z_train)

    z_pred = lasso.predict(X_test)
    r2 = R2(z_test, z_pred)
    mse = mean_squared_error(z_test, z_pred)

    print(f"\nLASSO REGRESSION WITH 80% TRAINING AND 20% TEST DATA:")
    print(f"Mean Squared Error: {mse}")
    print(f"R2-score: {r2}")

    # ---Cross validation to determine optimal lambda---
    n_lambdas = 500 # Number of lambdas
    k = 5           # Number of folds in the cross validation
    d = 10           # Degree of polynomial
    lambdas = np.logspace(-4,4, n_lambdas)      # Lambdas
    mse_scores = np.zeros((n_lambdas, k))       # Matrix to save the MSE-scores in

    for i in range(n_lambdas):
        #mse, train_err, r2_score = k_Cross_Validation(x1, y1, z1, k=k, reg_method='Lasso', lmb=lambdas[i])
        #mse_scores[i,:] = mse

        # Do CV
        model = Lasso(alpha=lambdas[i], max_iter=10e5, tol=0.001, fit_intercept=True, normalize=True)
        j=0
        for indexes in k_folds(n, k=k):
            x_test = x1[indexes]
            y_test = y1[indexes]
            z_test = z1[indexes]
            x_train = x1[np.delete(np.arange(n), indexes)]
            y_train = y1[np.delete(np.arange(n), indexes)]
            z_train = z1[np.delete(np.arange(n), indexes)]

            X_test = CreateDesignMatrix_X(x_test, y_test, d=d)
            X_train = CreateDesignMatrix_X(x_train, y_train, d=d)

            X_test = X_test[:,1:]
            X_train = X_train[:,1:]

            fit = model.fit(X_train, z_train)
            z_pred_train = fit.predict(X_train)
            z_pred_test = fit.predict(X_test)

            #mse_scores[i,j] = np.sum((z_pred_test - z_test)**2)/np.size(z_pred_test)
            mse_scores[i,j] = MSE(z_pred_test, z_test)
            j+=1


    estimated_mse_CV = np.mean(mse_scores, axis=1) 

    plt.plot(np.log10(lambdas), estimated_mse_CV, label='mse')
    plt.xlabel('log10(lambda)')
    plt.ylabel('MSE')
    plt.title(f"MSE vs. lambda (Lasso regression with degree {d})")
    plt.legend()
    plt.show()
    """

    return



"""
Other stuff
"""

def mse_lambda_plot():

    n_lambdas = 500
    lambdas = np.logspace(-5,6,n_lambdas)
    d = 5
    k = 5

    mse_scores = []
    mse_train_scores = []

    mse_scores_lasso = []
    mse_train_scores_lasso = []

    r2_ridge = []
    r2_lasso = []
    r2_ols = []

    test_err, train_err, r2 = k_Cross_Validation(x1, y1, z1, k=k, d=d)
    r2_lin = test_err


    for lmb in lambdas:
        r2_ols.append(r2_lin)

        test_err, train_err, r2 = k_Cross_Validation(x1, y1, z1, k=k, d=d, reg_method='Ridge', lmb=lmb)
        mse_scores.append(test_err)
        mse_train_scores.append(train_err)
        r2_ridge.append(r2)
        test_err, train_err, r2 = k_Cross_Validation(x1, y1, z1, k=k, d=d, reg_method='Lasso', lmb=lmb)
        mse_scores_lasso.append(test_err)
        mse_train_scores_lasso.append(train_err)
        r2_lasso.append(r2)

    
    plt.plot(np.log10(lambdas), mse_scores, label='MSE test Ridge')
    plt.plot(np.log10(lambdas), mse_train_scores, label='MSE train Ridge')
    plt.plot(np.log10(lambdas), mse_scores_lasso, label="MSE test Lasso")
    plt.plot(np.log10(lambdas), mse_train_scores_lasso, label='MSE train Lasso')
    plt.xlabel('log10(lambda)')
    plt.ylabel('MSE')
    plt.legend()
    save_fig('mse-lambda-ridge-lasso-plot')
    plt.show()

    plt.plot(np.log(lambdas), r2_ridge, label='R2-score ridge')
    plt.plot(np.log(lambdas), r2_lasso, label='R2-score lasso')
    plt.plot(np.log(lambdas), r2_ols, label="R2-score linear reg")
    plt.xlabel('log10(lambda)')
    plt.ylabel('R2-score')
    plt.legend()
    save_fig('r2-lambda-ridge-lasso-plot')
    plt.show()

    return



part_a()
#part_b()
#part_c()
#part_d()
#part_e()
#mse_lambda_plot()

