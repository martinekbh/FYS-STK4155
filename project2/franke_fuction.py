# General imports
import numpy as np
from sklearn.neural_network import MLPRegressor

# Import own classes and functions
from NeuralNetwork import NeuralNetwork
from own_code import *


# ---MAKE DATA---
k = 100                       # Number of points on each axis
x = np.arange(0, 1, 1/k)     # Numbers on x-axis
y = np.arange(0, 1, 1/k)     # Numbers on y-axis
x, y = np.meshgrid(x,y)      # Create meshgrid of x and y axes
z = FrankeFunction(x,y)      # The z-values

x1 = np.ravel(x)            # Flatten to vector
y1 = np.ravel(y)            # Flatten to vector
n = len(x1)                 # Number of observations (n=k*k)
np.random.seed(1001)        # Set seed for reproducability
z1 = np.ravel(z) + np.random.normal(0, .1, n)    # Add noise if wanted

# Plot Franke Function
make3Dplot(x,y,z, name="franke_func", show=False)

# Set up training data
test_size = 0.3
test_inds, train_inds = test_train_index(n, test_size=test_size)
x_train = x1[train_inds]
x_test = x1[test_inds]
y_train = y1[train_inds]
y_test = y1[test_inds]
z_train = z1[train_inds]
z_test = z1[test_inds]
X_train = CreateDesignMatrix_X(x_train, y_train, d=1)
X_test = CreateDesignMatrix_X(x_test, y_test, d=1)

# Scale data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


def NeuralNet_test():
    # Create Neural Network
    eta = 1e-1                   # Learning rate
    lmbd = 0                   # Regression parameter
    epochs = int(1e+2)               # Number of epochs
    batch_size = int(n/50)      # Batch size = (number of observations) / (number of batches)
    n_layers = 5                # Number of hidden layers
    n_hidden_neurons = [50]*n_layers      # Number of neurons in the hidden layers
    activation = 'leakyReLU'

    nn = NeuralNetwork(X_train, z_train, n_layers=n_layers, activation=activation,
                        n_hidden_neurons=n_hidden_neurons, epochs=epochs,
                        batch_size=batch_size,
                        eta=eta, lmbd=lmbd, problem='reg', seed=1)

    nn.train()
    z_pred = nn.predict(X_test)

    # Performance
    r2score = R2(z_test, z_pred)
    mse = MSE(z_test, z_pred)
    print("R2-score: ", r2score)
    print("MSE: ", mse)
    # Plot
    make3Dplot(x_test, y_test, z_pred.ravel(), name=f"ownNNreg_{n_layers}layers_{activation}", show=True)


    # --- SCIKIT LEARN NEURAL NETWORK TEST (for comparison)---
    eta = 1e-2
    scikitNN = MLPRegressor(hidden_layer_sizes=n_hidden_neurons, alpha=lmbd,
                                learning_rate_init=eta, max_iter=epochs)
    scikitNN.fit(X_train, z_train)
    print("SCIKIT learn MLPRegressor:")
    print(scikitNN.score(X_test, z_test))
    print(scikitNN.batch_size)
    make3Dplot(x_test, y_test, scikitNN.predict(X_test).ravel(),
            name=f"scikitNNreg_{len(n_hidden_neurons)}hiddenlayers_ReLU", show=True)

def NeuralNet_find_opt_params():


    eta = 1e-1                   # Learning rate
    lmbd = 0                   # Regression parameter
    epochs = int(1e+2)               # Number of epochs
    batch_size = int(n/50)      # Batch size = (number of observations) / (number of batches)
    n_layers = 5                # Number of hidden layers
    n_hidden_neurons = [50]*n_layers      # Number of neurons in the hidden layers
    activation = 'leakyReLU'

    #batch_size = 50
    #n_layers = 3
    #n_hidden_neurons = [50,40,30]

    """
    # DO GRID SEARCH to find optinal FFNN hyperparameters lmbd and eta
    print(f"\nPerforming grid test to find optimal learning rate and lambda for the Neural Network:")
    eta_vals = np.logspace(-5,1,7)
    lmbd_vals = np.logspace(-5,1,7)

    nn_grid = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
    #nn_grid = []

    mse_scores = np.zeros((len(eta_vals), len(lmbd_vals)))
    R2_scores = np.zeros((len(eta_vals), len(lmbd_vals)))
    for i, eta in enumerate(eta_vals):
        #nn_grid.append([])
        for j,lmbd in enumerate(lmbd_vals):

            print(f"eta_vals[{i}], lmbd_vals[{j}]")
            nn = NeuralNetwork(X_train, z_train, eta=eta, lmbd=lmbd,
                    epochs=epochs, batch_size=batch_size, n_layers=n_layers,
                    n_hidden_neurons=n_hidden_neurons, problem='reg', activation=activation)
            nn.train()                          # Train network with train data
            #nn_grid[i].append(nn)               # Insert trained network in grid
            nn_grid[i][j] = nn
            test_predict = nn.predict(X_test)   # Use network to predict test data
            mse = MSE(z_test, test_predict)     # Calculate MSE
            mse_scores[i][j] = mse              # Insert mse in grid
            r2 = R2(z_test, test_predict)       # Calculate R2-score
            R2_scores[i][j] = r2                # Insert R2 in grid

    print(f"Minimum MSE: {np.nanmin(mse_scores)}")
    opt_eta_index, opt_lmbd_index = np.where(mse_scores == np.nanmin(mse_scores))
    opt_eta = eta_vals[opt_eta_index]
    opt_lmbd = lmbd_vals[opt_lmbd_index]
    print(f"Obtained with parameters:")
    print(f"Learning rate={opt_eta}, Lambda={opt_lmbd}")
    print(f"Test: mse={mse_scores[opt_eta_index, opt_lmbd_index][0]}")
    print(f"      r2={R2_scores[opt_eta_index, opt_lmbd_index][0]}")


    # PLOT accuracy vs. learning rate and lambda
    import matplotlib.pyplot as plt
    xmax = np.log10(eta_vals[-1])
    xmin = np.log10(eta_vals[0])
    ymax = np.log10(lmbd_vals[-1])
    ymin = np.log10(lmbd_vals[0])
    fig, ax = plt.subplots()
    ax.matshow(mse_scores, cmap=plt.cm.Wistia_r,
                extent = [xmin-0.5, xmax+0.5, ymax+0.5, ymin-0.5],
                interpolation=None, aspect='auto', origin='upper')
    for i in range(len(eta_vals)):
        for j in range(len(lmbd_vals)):
            c = mse_scores[i,j]
            #c = *round(c,3)
            ax.text(np.log10(lmbd_vals[j]), np.log10(eta_vals[i]), f'{c:.2E}', va='center', ha='center')

    plt.ylabel("log10(learning rate)")
    plt.xlabel("log10(lambda)")
    save_fig("NN_mse_map_frankefunc2", extension='pdf')
    plt.show()
    """

    """
    # Plot accuracy vs. learning rate for different lambas
    plt.figure()
    for l in range(len(lmbd_vals)):
        plt.plot(eta_vals, mse_scores[:,l], label=f'lmbd={lmbd_vals[l]:.1E}')

    plt.legend()
    plt.xlabel("log10(learning rate)")
    plt.ylabel("mse")
    save_fig("NN_mse_vs_eta", extension='pdf')
    #plt.show()
    """

    # Second grid search for further tuning of the parameters
    print(f"\nPerforming second grid test to find optimal learning rate and lambda for the Neural Network:")
    #eta_vals = np.logspace(-2,1,5)
    #lmbd_vals = np.linspace(-4,-2,5)
    eta_vals = np.logspace(-1,0.4,7)
    lmbd_vals = np.logspace(-5.5, -4.5,7)

    nn_grid = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
    mse_scores = np.zeros((len(eta_vals), len(lmbd_vals)))
    R2_scores = np.zeros((len(eta_vals), len(lmbd_vals)))
    for i, eta in enumerate(eta_vals):
        for j,lmbd in enumerate(lmbd_vals):

            print(f"eta_vals[{i}], lmbd_vals[{j}]")
            nn = NeuralNetwork(X_train, z_train, eta=eta, lmbd=lmbd,
                    epochs=epochs, batch_size=batch_size, n_layers=n_layers,
                    n_hidden_neurons=n_hidden_neurons, problem='reg', activation=activation)
            nn.train()                          # Train network with train data
            nn_grid[i][j] = nn
            test_predict = nn.predict(X_test)   # Use network to predict test data
            mse = MSE(z_test, test_predict)     # Calculate MSE
            mse_scores[i][j] = mse              # Insert mse in grid
            r2 = R2(z_test, test_predict)       # Calculate R2-score
            R2_scores[i][j] = r2                # Insert R2 in grid

    print(mse_scores)
    print(f"\nMinimum MSE: {np.nanmin(mse_scores)}")
    opt_eta_index, opt_lmbd_index = np.where(mse_scores == np.nanmin(mse_scores))
    opt_eta = eta_vals[opt_eta_index]
    opt_lmbd = lmbd_vals[opt_lmbd_index]
    print(f"Obtained with parameters:")
    print(f"Learning rate={opt_eta}, Lambda={opt_lmbd}")
    print(f"Test: mse={mse_scores[opt_eta_index, opt_lmbd_index][0]}")
    print(f"      r2={R2_scores[opt_eta_index, opt_lmbd_index][0]}")

    # Finally, plot estimated Frankes function using the best NN
    print(f"\nTesting results of the best Neural network again:")
    bestNN = nn_grid[opt_eta_index, opt_lmbd_index][0]
    z_pred = bestNN.predict(X_test)
    r2score = R2(z_test, z_pred)
    mse = MSE(z_test, z_pred)
    print("MSE: ", mse)
    print("R2-score: ", r2score)
    make3Dplot(x_test, y_test, z_pred.ravel(), name=f"ownNNreg_optimalparams", show=True)




# ---Linear Regression---
def linreg_find_opt_degree():
    # ---MAKE PLOT OF ERROR VS. DEGREE OF POLYNOMIAL---
    print("\nLINEAR REGRESSION ANALYSIS")
    maxdegree = 30
    degrees = np.arange(1, maxdegree+1)

    test_err_results = []
    train_err_results = []
    r2_score_results = []

    print(f"Testing which degree from 1 to {maxdegree} gives lowest MSE using 5-fold CV...")
    for deg in degrees:
        print(f"degree {deg}")
        test_err, train_err, r2 = k_Cross_Validation(x1, y1, z1, d=deg)
        test_err_results.append(test_err)
        train_err_results.append(train_err)
        r2_score_results.append(r2)


    # Plot test-error and train-error
    plt.figure()
    plt.plot(degrees, test_err_results, 'k', label='Test MSE')
    plt.plot(degrees, train_err_results, 'b', label='Train MSE')
    plt.legend()
    plt.xlabel('degree of polynomial')
    plt.ylabel('error')
    save_fig('Franke_linreg_train_test_error_plot')
    #plt.title('training and test error vs. polynomial degree')
    plt.show()

    # ---DO LINEAR REGRESSION FOR OPTIMAL DEGREE d---
    min_index = np.where(test_err_results == min(test_err_results))
    deg_ind  = tuple(i.item() for i in min_index)
    opt_degree = degrees[deg_ind]
    print("Optimal degree: ", opt_degree)

    test_err, train_err, r2 = k_Cross_Validation(x1, y1, z1, d=opt_degree, reg_method='Linear')
    print(f"\nLINEAR REGRESSION (with 5-fold CV) USING OPTIMAL DEGREE {opt_degree}:")
    print(f"MSE = {test_err}\nR2 = {r2}")

def ridge_find_opt_params():
    print(f"\nRIDGE REGRESSION ANALYSIS")
    #np.random.seed(1001)
    n_lambdas = 10
    #lambdas = np.logspace(-8,0, n_lambdas)
    lambdas = np.logspace(-12, -7, n_lambdas)
    maxdegree = 35
    #degrees = np.arange(1, maxdegree+1)
    degrees = np.arange(22, 35)
    k = 5

    mse_scores = np.zeros((len(degrees), n_lambdas)) # Matrix to save the mse-scores

    i=0
    for deg in degrees:
        j=0
        print(f"degree {deg}")
        for lmb in lambdas:
            test_err, train_err, r2 = k_Cross_Validation(x1, y1, z1, k=k, d=deg, reg_method='Ridge', lmb=lmb)
            mse_scores[i,j] = test_err
            j += 1
        i += 1

    #deg_ind, lmb_ind = np.where(mse_scores == np.nanmin(mse_scores))
    #opt_degree = degrees[deg_ind]
    #opt_lambda = lambdas[lmb_ind]

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
    save_fig('Franke_ridge_colormap2')
    plt.show()

    # Best results
    test_err, train_err, r2 = k_Cross_Validation(x1, y1, z1, d=opt_degree, lmb=opt_lambda, reg_method='Ridge')
    print(f"\nRIDGE REGRESSION (with 5-fold CV) USING OPTIMAL DEGREE {opt_degree} AND LAMBDA {opt_lambda}:")
    print(f"MSE = {test_err}\nR2 = {r2}")

    return

def lasso_find_opt_params():
    print(f"\nLASSO ANALYSIS")
    n_lambdas = 10
    #lambdas = np.logspace(-4,0, n_lambdas)
    lambdas = np.logspace(-6, -4, n_lambdas)
    maxdegree = 30
    #degrees = np.arange(1, maxdegree+1)
    degrees = np.arange(24, 31)
    k = 5

    mse_scores = np.zeros((len(degrees), n_lambdas)) # Matrix to save the mse-scores

    i=0
    for deg in degrees:
        j=0
        print(f"degree {deg}")
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
    save_fig('Franke_lasso_colormap2')
    plt.show()

    # Best results
    test_err, train_err, r2 = k_Cross_Validation(x1, y1, z1, d=opt_degree, lmb=opt_lambda, reg_method='Lasso')
    print(f"\nLASSO REGRESSION (with 5-fold CV) USING OPTIMAL DEGREE {opt_degree} AND LAMBDA {opt_lambda}:")
    print(f"MSE = {test_err}\nR2 = {r2}")


#NeuralNet()
#NeuralNet_find_opt_params()
#linreg_find_opt_degree()
#ridge_find_opt_params()
lasso_find_opt_params()

