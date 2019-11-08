"""
Module that contains all relevant functions of our own making.
"""
# General imports
import numpy as np
import pandas as pd
#from random import random, seed
import os

# Import scikit regression tools
from sklearn.model_selection import train_test_split, cross_validate, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge, SGDClassifier
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures,StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer

# Import plotting tools
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter



class data_manager:
    def __init__(self, X, y, predictors = 0):
            # Features and targets
        self.X = X
        self.y = y

        self.n = len(self.y)         # Number of observations
        self.p = np.shape(self.X)[1] # Number of explanatory variables

        # Whacky method to get predictor names correct
        if isinstance(predictors, int):
            self.predictors = np.arange(1,self.p+1)

        else:
            self.predictors = predictors


    def split_data(self, Testsize = 0.33, seed = None):
        """
            input:
                test_size: The size of the test data which is going to be
                validated against train data with size 1 - test_size.
                Default size 1/5.
                seed:

            return:
                test_data: Ndarrray
                train_data: Ndarray

        """
        test_index, train_index = self.test_train_index(self.n, test_size = Testsize, seed=seed)

        X_train = self.X[train_index]
        X_test = self.X[test_index]

        self.y_train = self.y[train_index]
        self.y_test  = self.y[test_index]

        # Scale data
        self.X_train, self.X_test = self.scale_data(X_train, X_test)


        return self.X_test, self.X_train, self.y_test, self.y_train

    def scale_data(self, X_train, X_test):
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_test  = scaler.transform(X_test)
        X_train = scaler.transform(X_train)
        return X_train, X_test


    def create_design_matrix(self, regression_type):
        if regression_type == "log":
            one_vector = np.ones(np.shape(self.X[:,0])).reshape(len(X[:,0]),1)
            A = np.concatenate((one_vector,self.X), axis = 1)
            self.A  = A
            return self.A


    def test_train_index(self, n, test_size=0.33, seed=None):
        """
        Returns indexes for randomly selected train set and test set.
        args:
            n is an int specifying the number of indexes
            test_size is between 0 and 1, and is the size of the test set (as a ratio)
        """
        indexes = np.arange(n)

        if seed != None:
            np.random.seed(seed)
        np.random.shuffle(indexes)

        n_test = int(n*test_size)
        test_indexes = indexes[0:n_test]
        train_indexes = indexes[n_test:]

        return test_indexes, train_indexes


class solver:
    def __init__(self, credit_card_object, seed = None):
        self.seed = seed
        if seed != None:
            np.random.seed(seed)

        self.cred = credit_card_object

        # the split_data() function also scales the data
        self.Xtest, self.Xtrain, self.ytest, self.ytrain = self.cred.split_data(seed=seed)
        self.predictors = self.cred.predictors  # Names of the predictors

        

    def minibatch(self, M):
        #print(self.Xtrain)
        matrix_A = self.Xtrain
        matrix_length = len(matrix_A[:,0])
        #print("---------------------")
        if M > matrix_length:
            raise ValueError("The size of batches cannot be larger than the matrix length")
        else:
            batch_length = int(matrix_length/M)

        mini_row_matrix = []
        for i in range(0, matrix_length, batch_length):
            #print(i)
            mini_row_matrix.append(matrix_A[i-1:M*batch_length,:])

        #print(mini_row_matrix[0])


    #def SGD_integrator(self, Niter, seed = None):

    def sgd(self, n_epochs, n_minibatches=None, print_coeffs=False):
        Xtrain = self.Xtrain; Xtest = self.Xtest; ytrain = self.ytrain; ytest = self.ytest
        n = len(Xtrain)

        if n_minibatches == None: # Default number of minibatches is n
            n_minibatches = n
            batch_size = 1

        else:
            batch_size = int(n / n_minibatches)

        beta = np.random.randn(len(Xtrain[0]), 1) # why random?

        for epoch in range(n_epochs):   # epoch

            j = 1
            for i in range(n_minibatches):          # minibatches
                random_index = np.random.randint(n_minibatches)

                xi = Xtrain[random_index * batch_size: random_index*batch_size + batch_size]
                yi = ytrain[random_index * batch_size: random_index*batch_size + batch_size]
                yi = yi.reshape((batch_size, 1))

                p = 1/(1 + np.exp(-xi @ beta))
                gradient = -xi.T @ (yi - p) 
                l = self.learning_schedule(epoch*n_minibatches + i)
                beta = beta - l * gradient
                self.beta = beta

        if print_coeffs == True:
            #beta_coeff = sGD_classifier.coef_[0]    # This is a nested list for some reason
            #beta_intercept = sGD_classifier.intercept_[0]
            self.print_coeff_table(beta[0], beta[1:])

        return beta

    def learning_schedule(self, t, t0=5, t1=50):
        return t0/(t+t1)

    def predict(self, x, beta=None):
        if beta == None:
            beta = self.beta

        pred = np.round(1/(1 + np.exp(-x@beta))).ravel()
        return pred

    def stocastic_gradient_descent(self, Niter, print_coeffs = False):
        #if seed != None:
        #    np.random.seed(seed)

        sGD_classifier = SGDClassifier(loss = 'log',
                    penalty = 'l2', max_iter = Niter, eta0 = 0.001, fit_intercept = True,
                    shuffle = True, random_state = self.seed, n_iter_no_change = 5)
        sGD_classifier.fit(self.Xtrain,self.ytrain.ravel())

        ypred = sGD_classifier.predict(self.Xtest)

        # Scores
        R2_score = self.R2(self.ytest, ypred)
        mSE_score = self.MSE(self.ytest, ypred)
        accuracy = self.accuracy(self.ytest, ypred)
        print(f" MSE = {mSE_score}, R2-score = {R2_score}")
        print(f"accuracy = {accuracy}")


        if print_coeffs == True:
            beta_coeff = sGD_classifier.coef_[0]    # This is a nested list for some reason
            beta_intercept = sGD_classifier.intercept_[0]
            self.print_coeff_table(beta_coeff, beta_intercept)

    def print_coeff_table(self, beta_coeff, beta_intercept):
        from prettytable import PrettyTable
        table = PrettyTable()
        column_names = ["Predictor", "Coefficient"]
        table.add_column(column_names[0], self.predictors)
        table.add_column(column_names[1], beta_intercept+beta_coeff)
        print(table)

        return

    def R2(self, z, z_pred):
        """ Function to evaluate the R2-score """
        z = np.ravel(z)
        z_pred = np.ravel(z_pred)
        mean = (1/len(z))*sum(z)
        r2 = 1 - (sum((z-z_pred)**2)/sum((z - mean)**2))
        return r2

    def MSE(self, z, z_pred):
        """ Function to evaluate the Mean Squared Error """
        z = np.ravel(z)
        z_pred = np.ravel(z_pred)
        mse = (1/len(z))*np.sum((z-z_pred)**2)
        return mse

    def accuracy(self, y, y_pred):
        return np.sum(y == y_pred)/len(y)



class NeuralNetwork:
    def __init__(self, X, y, 
        n_layers = 1,
        n_hidden_neurons=[50], 
        n_categories = 2, 
        epochs=10, batch_size=100, 
        eta=0.1, lmbd=0.0, 
        seed = None):
        
        # data
        self.X = X
        self.y = y              # y must be the shape of OneHotEncoder
        self.X_full_data = X
        self.y_full_data = y

        # other vars
        self.n_inputs, self.n_features = X.shape
        self.n_layers = n_layers
        self.n_categories = n_categories
        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd

        # Set number of hidden neurons
        if n_hidden_neurons == None:
            self.n_hidden_neurons = [50]*n_layers
        elif isinstance(n_hidden_neurons, list) and len(n_hidden_neurons)==n_layers:
            self.n_hidden_neurons = n_hidden_neurons
        else:
            print(isinstance(n_hidden_neurons, list))
            print(len(n_hidden_neurons)==n_layers)
            print(n_layers)
            print(len(n_hidden_neurons))
            msg = "The arg n_hidden_neurons must be a list of integers," \
                    "and must have length equal to the arg n_layers."
            raise ValueError(msg)

        # Set seed
        if seed != None:
            np.random.seed(seed)

        self.create_biases_and_weights()

    def create_biases_and_weights(self):
        W = [np.random.randn(self.n_features, self.n_hidden_neurons[0])]
        b = [np.zeros((self.n_hidden_neurons[0], 1)) + 0.01]
        for l in range(1, self.n_layers):
            W.append( np.random.randn(self.n_hidden_neurons[l-1], self.n_hidden_neurons[l]) )
            b.append( np.zeros((self.n_hidden_neurons[l], 1)) + 0.01 )

        #self.hidden_bias = np.zeros((self.n_layers, self.n_hidden_neurons, 1)) + 0.01
        self.hidden_weights = W
        self.hidden_bias = b

        self.output_weights = np.random.randn(self.n_hidden_neurons[-1], self.n_categories)
        self.output_bias = np.zeros((self.n_categories, 1)) + 0.01

    def feed_forward(self):
        """ Feed forward for training """
        self.z_h = []
        self.a_h = []
        a = self.X

        for l in range(self.n_layers):
            #print(f"layer {l}")
            #z = (self.hidden_weights[l].T @ a_h) + self.hidden_bias[l]
            #self.z_h.append(z)
            z = np.matmul(a, self.hidden_weights[l]) + self.hidden_bias[l].T
            a = self.sigmoid(z)
            self.a_h.append(a)

        #self.z_o = (self.output_weights.T @ a_h) + self.output_bias.T
        self.z_o = np.matmul(a, self.output_weights) + self.output_bias.T
        exp_term = np.exp(self.z_o)
        self.probabilities = exp_term / (np.sum(exp_term, axis=1, keepdims=True))

    def feed_forward_out(self, X):
        """ Feed forward for output """
        a_h = X

        for l in range(self.n_layers):
            z = np.matmul(a_h, self.hidden_weights[l]) + self.hidden_bias[l].T
            a_h = self.sigmoid(z)

        z_o = np.matmul(a_h, self.output_weights) + self.output_bias.T
        exp_term = np.exp(z_o)
        probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
        return probabilities


    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def tanh(self, x):
        #return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
        return np.tanh(x)

    def back_propagation(self):
        a_L = self.probabilities
        error_output = a_L * (1 - a_L )*(a_L - self.y)
        #error_output = a_L * (1 - a_L )*(a_L - self.y.reshape(len(self.y), 1))
        a_h = self.a_h[-1]

        # The gradients of the outputs
        self.output_weights_gradient = np.matmul(a_h.T, error_output)
        self.output_bias_gradient = np.sum(error_output, axis=0)            # Is reshaped
        self.output_bias_gradient = self.output_bias_gradient.reshape(len(self.output_bias_gradient),1)

        # Make empty lists
        error_hidden = []
        self.hidden_weights_gradient = []
        self.hidden_bias_gradient = []
        
        # Calculate error and gradients of the hidden layers
        err = np.matmul(error_output, self.output_weights.T) * a_h * (1 - a_h)
        for l in range((self.n_layers-2), -1, -1):
            #print(f"l: {l}")
            error_hidden.insert(0,err) 
            self.hidden_weights_gradient.insert( 0, np.matmul(self.a_h[l].T, err) )
            self.hidden_bias_gradient.insert( 0, np.sum(err, axis=0).reshape(len(err[0]), 1) )

            err = np.matmul(err, self.hidden_weights[l+1].T) * self.a_h[l] * (1 - self.a_h[l]) 

        self.hidden_weights_gradient.insert( 0, np.matmul(self.X.T, err) )
        self.hidden_bias_gradient.insert( 0, np.sum(err, axis=0).reshape(len(err[0]),1) )

        # Regression parameter
        if self.lmbd > 0.0:
            self.output_weights_gradient += self.lmbd * self.output_weights
            for i in range(len(self.hidden_weights_gradient)): # Maybe -1 on the range?
                self.hidden_weights_gradient[i] += self.lmbd * self.hidden_weights[i]

        # Update the weights and biases 
        self.output_weights -= self.eta * self.output_weights_gradient
        self.output_bias -= self.eta * self.output_bias_gradient
        for i in range(len(self.hidden_weights)):
            self.hidden_weights[i] -= self.eta * self.hidden_weights_gradient[i]
            self.hidden_bias[i] -= self.eta * self.hidden_bias_gradient[i]


    def sgd(self, n_epochs, n_minibatches=None):
        Xtrain = self.Xtrain; Xtest = self.Xtest; ytrain = self.ytrain; ytest = self.ytest
        n = len(Xtrain)

        if n_minibatches == None: # Default number of minibatches is n
            n_minibatches = n
            batch_size = 1

        else:
            batch_size = int(n / n_minibatches)

        beta = np.random.randn(len(Xtrain[0]), 1) # why random?
        for epoch in range(n_epochs):   # epoch
            j = 1
            for i in range(n_minibatches):          # minibatches
                random_index = np.random.randint(n_minibatches)

                xi = Xtrain[random_index * batch_size: random_index*batch_size + batch_size]
                yi = ytrain[random_index * batch_size: random_index*batch_size + batch_size]
                yi = yi.reshape((batch_size, 1))

                p = 1/(1 + np.exp(-xi @ beta))
                gradient = -xi.T @ (yi - p) 
                l = self.learning_schedule(epoch*n_minibatches + i)
                beta = beta - l * gradient
                self.beta = beta

        return beta 

    def predict(self, X):
        probabilities = self.feed_forward_out(X)
        return np.argmax(probabilities, axis=1)

    def predict_probabilities(self, X):
        probabilities = self.feed_forward_out(X)
        return probabilities

    def train(self):
        data_indices = np.arange(self.n_inputs)
        
        for i in range(self.epochs):
            #print(f"epoch nr {i}")
            for j in range(self.iterations):
                #print(f"iteration nr {j}")

                # Pick datapoint with repacement:
                chosen_datapoints = np.random.choice(
                        data_indices, size=self.batch_size, replace=False)

                # minibatch training data
                self.X = self.X_full_data[chosen_datapoints]
                self.y = self.y_full_data[chosen_datapoints]

                self.feed_forward()
                #print("\n\nFeed forward done\n\n")
                self.back_propagation()
                #print("\n\nBack Propagation done\n\n")

def accuracy(y, pred):
    y = y.ravel()
    pred = pred.ravel()
    return np.sum(y == pred) / len(y)



if __name__== "__main__":
    # Import credit card data
    #cwd = os.getcwd() # Current working directory
    #filename = cwd + '/default of credit card clients.xls'
    #cred = data_manager("default of credit card clients.xls")

    # Import breast cancer data
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target
    analyze_data = data_manager(X,y)
    analyze_data.create_design_matrix('log')

    integrate = solver(analyze_data)
    integrate.minibatch(2)
    #integrate.stocastic_gradient_descent(100000)

    # Analyse data with logistic regression
    datapack = data_manager(X,y, predictors = cancer.feature_names)

    analyze = solver(datapack, seed = 1001)

    l_rate = 0.1
    n_epochs = 100
    beta = analyze.sgd(n_epochs, n_minibatches=30)
    # Martine: Jeg tror vi mangler intercept i beta?

    pred = analyze.predict(analyze.Xtest)
    acc = analyze.accuracy(analyze.ytest, pred)
    print(f"\n\nLogistic regression with {n_epochs} and {l_rate} learning rate")
    print(f"Accuracy: {acc}")

    # Analyze data with FFNN
    #X_train = analyze_data.X_train
    #X_test = analyze_data.X_test
    #y_train = analyze_data.y_train
    #y_test = analyze_data.y_test
    
    y = y.reshape(len(y), 1)
    """
    cancer_indices = [i for i, x in enumerate(y) if x == 1] 
    no_cancer_indices = [i for i, x in enumerate(y) if x == 0] 
    y_cancer = y[cancer_indices]
    y_no_cancer = y[no_cancer_indices]
    X_cancer = X[cancer_indices]
    X_no_cancer = X[no_cancer_indices]
    print("Cancer: ", len(y_cancer))
    print("No cancer: ", len(y_no_cancer))
    print(f"{len(y_cancer)*100/len(y):.2f}% of the people have cancer")

    X_cancer_train, X_cancer_test, y_cancer_train, y_cancer_test = train_test_split(X_cancer, y_cancer, test_size = 0.33, random_state=0)
    X_no_cancer_train, X_no_cancer_test, y_no_cancer_train, y_no_cancer_test = train_test_split(X_no_cancer, y_no_cancer, test_size = 0.33, random_state=0)

    X_train = np.concatenate((X_cancer_train, X_no_cancer_train), axis=0)
    X_test = np.concatenate((X_cancer_test, X_no_cancer_test), axis=0)
    y_train = np.concatenate((y_cancer_train, y_no_cancer_train), axis=0)
    y_test = np.concatenate((y_cancer_test, y_no_cancer_test), axis=0)
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)



    # Use One Hot Encoder on y_train
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(categories='auto')
    encoder.fit(y_train)
    y_train = encoder.transform(y_train).toarray()
    print(y_train.shape)

    print(f"\n\nFast-Forward Neural Network")
    """
    nn = NeuralNetwork(X_train, y_train, seed=1)
    pred = nn.predict(X_test)
    acc = accuracy(y_test, pred)
    print(f"Untrained accuracy: {acc}")
    nn.train()
    pred = nn.predict(X_test)
    acc = accuracy(y_test, pred)
    print(f"Trained accuracy: {acc}")
    """

    # DO GRID SEARCH to find optinal FFNN hyperparameters lmbd and eta 
    eta_vals = np.logspace(-8,3,12)
    lmbd_vals = np.logspace(-8,3,12)

    nn_grid = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
    epochs = 100
    batch_size = 100
    n_hidden_neurons = []
    n_layers = 5
    n_hidden_neurons = [50, 40, 30, 20, 10]

    acc_scores = np.zeros((len(eta_vals), len(lmbd_vals)))

    for i, eta in enumerate(eta_vals):
        for j,lmbd in enumerate(lmbd_vals):
            nn = NeuralNetwork(X_train, y_train, eta=eta, lmbd=lmbd, 
                    epochs=epochs, batch_size=batch_size, n_layers=n_layers, n_hidden_neurons=n_hidden_neurons)
            nn.train()
            nn_grid[i][j] = nn
            test_predict = nn.predict(X_test)
            #acc = np.sum( test_predict == y_test.ravel() )/len(y_test)
            acc = accuracy(y_test, test_predict)
            
            print(f"Learning rate = {eta}")
            print(f"Lambda = {lmbd}")
            print(f"Accuracy score on test set: {acc}\n")

            acc_scores[i][j] = acc


    print(f"Maximum accuracy: {np.max(acc_scores)}")
    opt_eta_index, opt_lmbd_index = np.where(acc_scores == np.max(acc_scores))
    opt_eta = eta_vals[opt_eta_index]
    opt_lmbd = lmbd_vals[opt_lmbd_index]
    print(f"Obtained with parameters:")
    print(f"Learning rate={opt_eta}, Lambda={opt_lmbd}")
    print(f"Test: acc={acc_scores[opt_eta_index, opt_lmbd_index]}")


    # PLOT
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




