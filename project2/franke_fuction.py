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
#xmax = x/(np.amax(x))       # To normalize the x-axis
#ymax = y/(np.amax(y))       # To normalize the y-axis
x, y = np.meshgrid(x,y)      # Create meshgrid of x and y axes
z = FrankeFunction(x,y)      # The z-values
print(np.min(z), np.max(z))
print('---------------')

x1 = np.ravel(x)            # Flatten to vector
y1 = np.ravel(y)            # Flatten to vector
n = len(x1)                 # Number of observations (n=k*k)
np.random.seed(1001)        # Set seed for reproducability
z1 = np.ravel(z) #+ np.random.normal(0, .1, n)    # Add noise if wanted

# Plot Franke Function
make3Dplot(x,y,z, name="franke_func", show=False)

# ---REGRESSION WITH NEURAL NETWORK---
# Test if Netral Network for regression works

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
#from sklearn.preprocessing import normalize
#z_train = normalize(z_train.reshape(-1,1))
#print(np.min(z_train), np.max(z_train))
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# Create Neural Network
eta = 1e-1                   # Learning rate
lmbd = 0                   # Regression parameter
epochs = int(1e+2)               # Number of epochs
batch_size = int(n/50)      # Batch size = (number of observations) / (number of batches)
n_layers = 5                # Number of hidden layers
n_hidden_neurons = [80]*n_layers      # Number of neurons in the hidden layers
activation = 'leakyReLU'
nn = NeuralNetwork(X_train, z_train, n_layers=n_layers, activation=activation,
                    n_hidden_neurons=n_hidden_neurons, epochs=epochs,
                    batch_size=batch_size,
                    eta=eta, lmbd=lmbd, problem='reg', seed=1)

nn.train()
z_pred = nn.predict(X_test)

r2score = R2(z_test, z_pred)
mse = MSE(z_test, z_pred)
print("R2-score: ", r2score)        # Negativ score???
print("MSE: ", mse)

print('\n', z_test[:20])
print(z_pred[:20])

# Plot
#make3Dplot(x_test, y_test, z_test, title="Franke function (test set)", show=False)
#make3Dplot(x_test, y_test, z_pred.ravel(), title="Neural Network (regression)", show=True)

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_trisurf(x_test, y_test, z_pred.ravel(), cmap=cm.coolwarm, linewidth = 0,
                        antialiased=False)
plt.xlabel("x-axis")
plt.ylabel("y-axis")
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()


# --- SCIKIT LEARN NEURAL NETWORK TEST ---
scikitNN = MLPRegressor(hidden_layer_sizes=n_hidden_neurons, alpha=lmbd, 
                            learning_rate_init=eta, max_iter=epochs)
scikitNN.fit(X_train, z_train)
print("SCIKIT learn MLPRegressor:")
print(scikitNN.score(X_test, z_test))
print(scikitNN.batch_size)

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_trisurf(x_test, y_test, scikitNN.predict(X_test).ravel(), cmap=cm.coolwarm, linewidth = 0,
                        antialiased=False)
plt.xlabel("x-axis")
plt.ylabel("y-axis")
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()



"""
# DO GRID SEARCH to find optinal FFNN hyperparameters lmbd and eta
print(f"\nPerforming grid test to find optimal learning rate and lambda for the Neural Network:")
eta_vals = np.logspace(-14,-5,10)
lmbd_vals = np.logspace(-1,8,10)

nn_grid = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
epochs = 100
batch_size = 50
n_layers = 3
n_hidden_neurons = [50,40,30]

mse_scores = np.zeros((len(eta_vals), len(lmbd_vals)))
for i, eta in enumerate(eta_vals):
    for j,lmbd in enumerate(lmbd_vals):
        nn = NeuralNetwork(X_train, y_train, eta=eta, lmbd=lmbd,
                epochs=epochs, batch_size=batch_size, n_layers=n_layers, 
                n_hidden_neurons=n_hidden_neurons, problem='reg', activation=activation)
        nn.train()
        nn_grid[i][j] = nn
        test_predict = nn.predict(X_test)
        mse = MSE(y_test, test_predict)

        #print(f"Learning rate = {eta}")
        #print(f"Lambda = {lmbd}")
        #print(f"Accuracy score on test set: {acc}\n")
        mse_scores[i][j] = mse


print(f"Minimum MSE: {np.min(mse_scores)}")
opt_eta_index, opt_lmbd_index = np.where(mse_scores == np.nanmin(mse_scores))
opt_eta = eta_vals[opt_eta_index]
opt_lmbd = lmbd_vals[opt_lmbd_index]
print(f"Obtained with parameters:")
print(f"Learning rate={opt_eta}, Lambda={opt_lmbd}")
print(f"Test: mse={mse_scores[opt_eta_index, opt_lmbd_index]}")


# PLOT accuracy vs. learning rate and lambda
import matplotlib.pyplot as plt
xmax = np.log10(eta_vals[-1])
xmin = np.log10(eta_vals[0])
ymax = np.log10(lmbd_vals[-1])
ymin = np.log10(lmbd_vals[0])
fig, ax = plt.subplots()
ax.matshow(mse_scores, cmap=plt.cm.Blues,
            extent = [xmin-0.5, xmax+0.5, ymax+0.5, ymin-0.5],
            interpolation=None, aspect='auto', origin='upper')
for i in range(len(eta_vals)):
    for j in range(len(lmbd_vals)):
        c = mse_scores[i,j]
        c = 100*round(c,3)
        ax.text(np.log10(lmbd_vals[j]), np.log10(eta_vals[i]), str(c), va='center', ha='center')

plt.ylabel("log10(learning rate)")
plt.xlabel("log10(lambda)")
save_fig("NN_mse_map_frankefunc", extension='pdf')
plt.show()
"""
