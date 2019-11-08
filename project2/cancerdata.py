import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LogisticRegression

# Import own code
from NeuralNetwork import NeuralNetwork

# Where to save the figures and data files
PROJECT_ROOT_DIR = "Results"
FIGURE_ID = "Results/FigureFiles"
DATA_ID = "DataFiles/"

def accuracy(y, pred):
    y = y.ravel()
    pred = pred.ravel()
    return np.sum(y == pred) / len(y)

# Load dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
y = y.reshape(len(y), 1)        # Make column vector

# Set up training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logreg with scikit
logreg = LogisticRegression(solver='liblinear')
logreg.fit(X_train, y_train.ravel())
print("Test set accuracy: {:f}".format(logreg.score(X_test,y_test.ravel())))
logreg.fit(X_train_scaled, y_train.ravel())
print("Test set accuracy scaled data: {:f}".format(logreg.score(X_test_scaled,y_test.ravel())))


# Use One Hot Encoder on y_train
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(categories='auto')
encoder.fit(y_train)
y_train = encoder.transform(y_train).toarray()

# Test Neural NeuralNetwork = logreg with 1 hidden node and layer?
eta = 0.1
lmbd = 0
epochs = 100
batchsize = int(len(y)/30)
Nhidden_layer = 1
nhidden_neuron = [1]
neural_net = NeuralNetwork(X_train, y_train,eta = eta, lmbd = lmbd,
epochs = epochs, batch_size = batchsize, n_layers=Nhidden_layer, n_hidden_neurons = nhidden_neuron, seed = 1)
neural_net.train()
pred = neural_net.predict(X_test)
accuracy_nn = accuracy(y_test,pred)

print(accuracy_nn)
exit()

# DO GRID SEARCH to find optinal FFNN hyperparameters lmbd and eta
eta_vals = np.logspace(-8,3,12)
lmbd_vals = np.logspace(-8,3,12)

nn_grid = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
epochs = 100
batch_size = 50
n_layers = 1
n_hidden_neurons = [1]

acc_scores = np.zeros((len(eta_vals), len(lmbd_vals)))
for i, eta in enumerate(eta_vals):
    for j,lmbd in enumerate(lmbd_vals):
        nn = NeuralNetwork(X_train, y_train, eta=eta, lmbd=lmbd,
                epochs=epochs, batch_size=batch_size, n_layers=n_layers, n_hidden_neurons=n_hidden_neurons)
        nn.train()
        nn_grid[i][j] = nn
        test_predict = nn.predict(X_test)
        acc = accuracy(y_test, test_predict)

        #print(f"Learning rate = {eta}")
        #print(f"Lambda = {lmbd}")
        #print(f"Accuracy score on test set: {acc}\n")
        acc_scores[i][j] = acc


print(f"Maximum accuracy: {np.max(acc_scores)}")
opt_eta_index, opt_lmbd_index = np.where(acc_scores == np.max(acc_scores))
opt_eta = eta_vals[opt_eta_index]
opt_lmbd = lmbd_vals[opt_lmbd_index]
print(f"Obtained with parameters:")
print(f"Learning rate={opt_eta}, Lambda={opt_lmbd}")
print(f"Test: acc={acc_scores[opt_eta_index, opt_lmbd_index]}")


# PLOT accuracy vs. learning rate and lambda
import matplotlib.pyplot as plt
xmax = np.log10(eta_vals[-1])
xmin = np.log10(eta_vals[0])
ymax = np.log10(lmbd_vals[-1])
ymin = np.log10(lmbd_vals[0])
fig, ax = plt.subplots()
ax.matshow(acc_scores, cmap=plt.cm.Blues,
            extent = [xmin-0.5, xmax+0.5, ymax+0.5, ymin-0.5],
            interpolation=None, aspect='auto', origin='upper')
for i in range(len(eta_vals)):
    for j in range(len(lmbd_vals)):
        c = acc_scores[i,j]
        c = 100*round(c,3)
        ax.text(np.log10(lmbd_vals[j]), np.log10(eta_vals[i]), str(c), va='center', ha='center')

plt.ylabel("log10(learning rate)")
plt.xlabel("log10(lambda)")
plt.show()
