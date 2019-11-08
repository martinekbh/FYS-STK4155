# General imports
import numpy as np

# Import own classes and functions
from NeuralNetwork import NeuralNetwork
from own_code import *


# ---MAKE DATA---
k = 20                       # Number of points on each axis
x = np.arange(0, 1, 1/k)     # Numbers on x-axis
y = np.arange(0, 1, 1/k)     # Numbers on y-axis
#xmax = x/(np.amax(x))       # To normalize the x-axis
#ymax = y/(np.amax(y))       # To normalize the y-axis
x, y = np.meshgrid(x,y)      # Create meshgrid of x and y axes
z = FrankeFunction(x,y)      # The z-values

x1 = np.ravel(x)            # Flatten to vector
y1 = np.ravel(y)            # Flatten to vector
n = len(x1)                 # Number of observations (n=k*k)
np.random.seed(1001)        # Set seed for reproducability
z1 = np.ravel(z) + np.random.normal(0, .25, n)    # Add noise if wanted

X = CreateDesignMatrix_X(x1, y1, d=1)

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

# Create Neural Network
eta = 0.1                   # Learning rate
lmbd = 0                    # Regression parameter
epochs = 100                # Number of epochs
batch_size = int(n/30)      # Batch size = (number of observations) / (number of batches)
n_layers = 1                # Number of hidden layers
n_hidden_neurons = [1]      # Number of neurons in the hidden layers
nn = NeuralNetwork(X_train, y_train, n_layers=n_layers,
                    n_hidden_neurons=n_hidden_neurons, epochs=epochs,
                    batch_size=batch_size,
                    eta=eta, lmbd=lmbd, problem='reg', seed=1)

# ARAM: Her skjer det en feil i back_propagation_regression()
nn.train()  # <--- feil her
z_pred = nn.predict(X_test)
acc = accuracy(z_test, z_pred)
print(acc)
