import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

# Import own code
from NeuralNetwork import NeuralNetwork
from LogReg import LogReg
from own_code import *

# Where to save the figures and data files
PROJECT_ROOT_DIR = "Results"
FIGURE_ID = "Results/FigureFiles"
DATA_ID = "DataFiles/"
"""
def accuracy(y, pred):
    y = y.ravel()
    pred = pred.ravel()
    return np.sum(y == pred) / len(y)
"""

# Load dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
y = y.reshape(len(y), 1)        # Make column vector

# Set up training data
seed = 0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# Scale data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---LOGREG WITH SCIKIT (scaled and not scaled)---
print("Scikit logistic regression results:")
logreg = LogisticRegression(solver='liblinear')
logreg.fit(X_train, y_train.ravel())
print("Test set accuracy: {:f}".format(logreg.score(X_test,y_test.ravel())))
logreg.fit(X_train_scaled, y_train.ravel())
print("Test set accuracy scaled data: {:f}".format(logreg.score(X_test_scaled,y_test.ravel())))

# Let X_train and X_test be scaled
X_train = X_train_scaled
X_test = X_test_scaled

# ---LOGREG WITH OWN CLASS---
# Add intercept column to the X-data
one_vector = np.ones((len(y_train),1))
X_train1 = np.concatenate((one_vector, X_train), axis = 1)
one_vector = np.ones((len(y_test),1))
X_test1 = np.concatenate((one_vector, X_test), axis = 1)

# Do logistic regression
logreg = LogReg(X_train1, y_train)
logreg.sgd(n_epochs=100, n_minibatches=30)
pred = logreg.predict(X_test1)
acc = accuracy(y_test, pred)
print(f"Test set accuracy with own logreg code: {acc}")

def LogReg_optimize_n_minibatches(epochs = 100, X=X, y=y):
    print("\nOptimizing number of minibatches for logistic regression...")
    # Insert column of 1's first
    one_vector = np.ones((len(y),1))
    X = np.concatenate((one_vector, X), axis = 1)
    #one_vector = np.ones((len(y_test),1))
    #X_test1 = np.concatenate((one_vector, X_test), axis = 1)
    #X_train = X_train1
    #X_test = X_test1

    iterations = 10
    plt.figure()
    n_minibatches = np.arange(1,150, 10)
    for iteration in range(iterations):
        print(f"iteration {iteration}")
        acc_scores = np.zeros(len(n_minibatches))
        acc_train_scores = np.zeros(len(n_minibatches))
        for i, n in enumerate(n_minibatches):
            #logreg = LogReg(X_train, y_train)
            #logreg.sgd(n_epochs=epochs, n_minibatches=n)
            #pred = logreg.predict(X_test)
            #acc = accuracy(y_test, pred)
            acc, train_acc = k_Cross_Validation_logreg(X, y, epochs=epochs, n_minibatches=n)
            acc_scores[i] = acc
            acc_train_scores[i] = train_acc
        plt.plot(n_minibatches, acc_scores, 'tab:blue')
        plt.plot(n_minibatches, acc_train_scores, 'tab:red')

    plt.xlabel("Number of minibatches in SGD")
    plt.ylabel("Accuracy score")
    #plt.ylim((0.8, 1))
    plt.legend(['Test set', 'Train set'])
    save_fig("LogRegcancer_accuracy_vs_n_minibatches")
    plt.show()

    opt_index = np.where(acc_scores == np.nanmax(acc_scores))
    opt_n_minibatches = n_minibatches[opt_index]
    return opt_n_minibatches

def LogReg_optimize_epochs(n_minibatches = 30, X=X, y=y):
    print("\nOptimizing number of epochs for logistic regression...")
    one_vector = np.ones((len(y),1))
    X = np.concatenate((one_vector, X), axis = 1)
    #one_vector = np.ones((len(y_train),1))
    #X_train1 = np.concatenate((one_vector, X_train), axis = 1)
    #one_vector = np.ones((len(y_test),1))
    #X_test1 = np.concatenate((one_vector, X_test), axis = 1)
    #X_train = X_train1
    #X_test = X_test1

    iterations = 10
    epochs = np.arange(1, 202, 10) 
    plt.figure()
    for iteration in range(iterations):
        print(f"iteration {iteration}")
        acc_scores = np.zeros(len(epochs))
        acc_train_scores = np.zeros(len(epochs))
        for i, ep in enumerate(epochs):
            #logreg = LogReg(X_train, y_train)
            #logreg.sgd(n_epochs=ep, n_minibatches=n_minibatches)
            #pred = logreg.predict(X_test)
            #acc = accuracy(y_test, pred)
            acc, train_acc = k_Cross_Validation_logreg(X, y, epochs=ep, n_minibatches=n_minibatches)
            acc_scores[i] = acc
            acc_train_scores[i] = train_acc    
        plt.plot(epochs, acc_scores, 'tab:blue')
        plt.plot(epochs, acc_train_scores, 'tab:red')
    plt.xlabel("Number of epochs")
    plt.ylabel("Accuracy score")
    plt.legend(["Test data","Train data"])
    save_fig("LogRegcancer_accuracy_vs_epochs")
    plt.show()

    opt_index = np.where(acc_scores == np.nanmax(acc_scores))
    opt_epochs = epochs[opt_index]
    return opt_epochs

def LogReg_with_opt_params(epochs, n_minibatches, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test):
    one_vector = np.ones((len(y_train),1))
    X_train1 = np.concatenate((one_vector, X_train), axis = 1)
    one_vector = np.ones((len(y_test),1))
    X_test1 = np.concatenate((one_vector, X_test), axis = 1)
    X_train = X_train1
    X_test = X_test1

    logreg = LogReg(X_train, y_train)
    logreg.sgd(n_epochs=epochs, n_minibatches=n_minibatches)
    pred = logreg.predict(X_test)
    acc = accuracy(y_test, pred)
    train_acc = accuracy(y_train, logreg.predict(X_train))
    #acc, train_acc = k_Cross_Validation_logreg(X, y, epochs=epochs, n_minibatches = n_minibatches)
    print(f"\nDoing logistic regression with optimal parameters")
    print(f"Number of epochs: {epochs}")
    print(f"Number of minibatches: {n_minibatches}")
    print(f"Accuracy score on test set: {acc}")
    print(f"Accuracy score on train set: {train_acc}")
    return acc


def NeuralNet_test(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test):
    # ---NEURAL NETWORK (OWN CLASS)---
    # Use One Hot Encoder on y_train
    print(f"Testing the Neural Network with some chosen parameters")
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(categories='auto')
    encoder.fit(y_train)
    y_train = encoder.transform(y_train).toarray()

    # Test Neural NeuralNetwork = logreg with 1 hidden node and layer?
    eta = 0.1
    lmbd = 0
    epochs = 100
    batchsize = int(len(y)/30)
    #batchsize = 50
    Nhidden_layer = 3
    nhidden_neuron = [50,40,30]
    neural_net = NeuralNetwork(X_train, y_train,eta = eta, lmbd = lmbd,
                    epochs = epochs, batch_size = batchsize, n_layers=Nhidden_layer, 
                    n_hidden_neurons = nhidden_neuron, seed = 1)
    neural_net.train()
    pred = neural_net.predict(X_test)
    accuracy_nn = accuracy(y_test,pred)
    print(f"epochs={epochs}")
    print(f"eta={eta}")
    print(f"lmbd={lmbd}")
    print(f"batchsize={batchsize}")
    print(f"Number of hidden layers: {Nhidden_layer}")
    print(f"Hidden neurons: {nhidden_neuron}")
    print(f"Test set accuracy with own Neural Network code: {accuracy_nn}")


def NeuralNet_find_optimal_eta_lmbd():
    # DO GRID SEARCH to find optinal FFNN hyperparameters lmbd and eta
    print(f"\nPerforming grid test to find optimal learning rate and lambda for the Neural Network:")
    eta_vals = np.logspace(-8,3,12)
    lmbd_vals = np.logspace(-18,2,21)

    nn_grid = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
    epochs = 100
    batch_size = 50
    n_layers = 3
    n_hidden_neurons = [50,40,30]

    acc_scores = np.zeros((len(eta_vals), len(lmbd_vals)))
    for i, eta in enumerate(eta_vals):
        for j,lmbd in enumerate(lmbd_vals):
            nn = NeuralNetwork(X_train, y_train, eta=eta, lmbd=lmbd,
                    epochs=epochs, batch_size=batch_size, n_layers=n_layers, n_hidden_neurons=n_hidden_neurons)
            nn.train()
            nn_grid[i][j] = nn
            test_predict = nn.predict(X_test)
            acc = accuracy(y_test, test_predict)
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
    ymax = np.log10(eta_vals[-1])
    ymin = np.log10(eta_vals[0])
    xmax = np.log10(lmbd_vals[-1])
    xmin = np.log10(lmbd_vals[0])
    fig, ax = plt.subplots()
    ax.matshow(10**acc_scores, cmap=plt.cm.summer,
                extent = [xmin-0.5, xmax+0.5, ymax+0.5, ymin-0.5],
                interpolation=None, aspect='auto', origin='upper')
    for i in range(len(eta_vals)):
        for j in range(len(lmbd_vals)):
            c = acc_scores[i,j]
            c = 100*round(c,3)
            ax.text(np.log10(lmbd_vals[j]), np.log10(eta_vals[i]), str(c), va='center', ha='center')

    plt.ylabel("log10(learning rate)")
    plt.xlabel("log10(lambda)")
    save_fig("NN_accuracy_map2")
    plt.show()

    opt_eta_index, opt_lmbd_index = np.where(acc_scores == np.nanmax(acc_scores))
    opt_eta = eta_vals[opt_eta_index]
    opt_lmbd = lmbd_vals[opt_lmbd_index]
    return opt_eta, opt_lmbd
    

def NeuralNet_optimize_epochs(opt_eta=0.1, opt_lmbd=1e-7, batch_size=50, n_layers=3, n_hidden_neurons=[50,40,30], X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test):
    print(f"\nOptimizing the number of epochs in the Neural Network...")
    batch_size = int(len(y)/30)
    opt_lmbd = 0

    # Use onehotencoder
    encoder = OneHotEncoder(categories='auto')
    encoder.fit(y_train)
    y_train_encode = encoder.transform(y_train).toarray()

    plt.figure()
    iterations = 10
    epochs = np.arange(1, 200, 10)
    for iteration in range(iterations):
        acc_scores = np.zeros(len(epochs))
        acc_train_scores = np.zeros(len(epochs))
        for i, ep in enumerate(epochs):
            nn = NeuralNetwork(X_train, y_train_encode, eta=opt_eta, lmbd=opt_lmbd,
                    epochs=ep, batch_size=batch_size, n_layers=n_layers, 
                    n_hidden_neurons=n_hidden_neurons)
            nn.train()
            test_predict = nn.predict(X_test)
            train_predict = nn.predict(X_train)
            acc = accuracy(y_test, test_predict)
            acc_train = accuracy(y_train, train_predict)

            acc_scores[i] = acc
            acc_train_scores[i] = acc_train
            print(acc_train)
            

        plt.plot(epochs, acc_scores, 'tab:blue')
        plt.plot(epochs, acc_train_scores, 'tab:red')
    plt.ylabel("Accuracy score")
    plt.xlabel("Number of epochs in the Neural Network")
    plt.legend(['Test data', 'Train data'])
    save_fig("NNcancer_acc_vs_epochs")
    plt.show()

    opt_index = np.where(acc_scores == np.nanmax(acc_scores))
    opt_epochs = epochs[opt_index]

    """
    # Plot accuracy vs. epochs for SCIKIT
    from sklearn.neural_network import MLPClassifier
    plt.figure()
    for iteration in range(iterations):
        acc_scores = np.zeros(len(epochs))
        for i, ep in enumerate(epochs):
            eta = 1e-2  # Sckikit may have other opt_eta?
            scikitNN = MLPClassifier(hidden_layer_sizes=n_hidden_neurons, alpha=opt_lmbd,
                                        learning_rate_init=opt_eta, max_iter=ep)
            scikitNN.fit(X_train, y_train)
            acc = scikitNN.score(X_test, y_test)
            acc_scores[i] = acc

        plt.plot(epochs, acc_scores, 'b')
    plt.ylabel("Accuracy score")
    plt.xlabel("Epochs")
    save_fig("scikitNNcancer_acc_vs_epochs")
    plt.show()
    """
    return opt_epochs


# --- Logsitic regression ---
#print("\nLogistic Regression Analysis:")
#opt_n_minibatches = LogReg_optimize_n_minibatches()
#print(f"Optimal number of minibatches: {opt_n_minibatches}")
#opt_epochs = LogReg_optimize_epochs()
#print(f"Optimal number of epochs: {opt_epochs}")
#acc = LogReg_with_opt_params(200, 50)

# --- Neural Network ---
print("\nNeural Network Analysis:")
NeuralNet_test()
#NeuralNet_optimize_eta_lmbd()
opt_epochs = NeuralNet_optimize_epochs()
print(f"Optimal number of epochs: {opt_epochs}")
