import tensorflow as tf 
import numpy as np 
from tensorflow import keras 
#from LogReg import LogReg
from own_code import *
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.linear_model import LogisticRegression
from sklearn. metrics import log_loss
import time

from tensorflow.keras.losses import categorical_crossentropy
loss = categorical_crossentropy     # Loss function

folder = "mnist_logreg"     # Folder for saving figure files

def mnist_data_for_logreg():
    mnist = tf.keras.datasets.mnist     # Load dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    train_shape = x_train.shape
    test_shape  = x_test.shape
    
    # Reshape and scale data
    x_train = x_train.ravel().reshape(train_shape[0], train_shape[1]*train_shape[2])/np.max(x_train)
    x_test =  x_test.ravel().reshape(test_shape[0], test_shape[1]*test_shape[2])/np.max(x_test)
    
    # Create "design matrix" with column of 1s first (for intercept)
    #one_vector = np.ones((len(y_train),1))
    #x_train1 = np.concatenate((one_vector, x_train), axis = 1)
    #one_vector = np.ones((len(y_test),1))
    #x_test1 = np.concatenate((one_vector, x_test), axis = 1)

    # Create one hot encoder
    #oneHot = OneHotEncoder()
    #oneHot.fit_transform()
    #encoded_ytrain = to_categorical(y_train, num_classes=10)
    #encoded_ytest = to_categorical(y_test, num_classes=10)
    #print(encoded_ytest[:10])
    #print(y_test[:10])

    return x_train, y_train, x_test, y_test
    #return x_train1, encoded_ytrain, x_test1, encoded_ytest

def LogReg(x_train, y_train, x_test, y_test):
    print("\nDoing logistic regression on the MNIST data....")
    start_time = time.time()

    # Fit logreg class
    logreg = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1e3)
    logreg.fit(x_train, y_train)

    # Make predictions
    pred_test = logreg.predict(x_test)
    pred_train = logreg.predict(x_train)

    stop_time = time.time()
    logreg_time = stop_time - start_time
    print(f"\nLogistic regression computation time: {logreg_time}s")

    # Calculate results
    test_acc = accuracy(y_test, pred_test)
    train_acc = accuracy(y_train, pred_train)
    test_loss = log_loss(y_test, to_categorical(pred_test))
    train_loss = log_loss(y_train, to_categorical(pred_train))
    # Print table of resutls
    print("Results:")
    print(f"\n     |  Train    |  Test")
    print("------------------------------------")
    print(f"Acc  |{train_acc:10.4f} |{test_acc:10.4f}")
    print(f"Loss |{train_loss:10.4f} |{test_loss:10.4f}\n")

    # Make confusion matrix of test set
    confr_matrix = confusionMatrix(pred_test, y_test, 
                        figurename="logreg_confusion_matrix", folder=folder, show=True)

    # Identify and plot the misclassified images
    imgShape = x_test.shape
    plotMisclassifiedImages(y_test, x_test.reshape(imgShape[0],28,28), pred_test, 
            figurename="sample_misclassified_images", folder=folder, rgb=False)

    return logreg


def LogReg_analyze_mnist(x_train, y_train, x_test, y_test, epochs, batch_size, n_image):
    n_train = n_image 
    n_test = int(n_train/100)
    n_minibatches = int(len(x_train)/batch_size)
   
    test_acc_list = []      # Test accuracy for each iteration
    train_acc_list = []     # Train accuracy for each iteration
    test_loss_list = []     # Test loss for each iteration
    train_loss_list = []    # Train loss for each iteration

    for i in range(1, epochs+1):  # Loop over number of epochs
        print(f"{i}/{epochs} max epochs")
        #logreg_analyzer = LogReg(x_train, y_train)
        #logreg_analyzer.sgd(n_epochs = i, n_minibatches = n_minibatches)
        logreg_analyzer = LogisticRegression(solver="lbfgs", multi_class="multinomial", max_iter=1e4)
        logreg_analyzer.fit(x_train, y_train)
        
        # Predict class of test set and store accuracy score
        pred_test = logreg_analyzer.predict(x_test)       
        test_acc_list.append(accuracy(y_test, pred_test))
        test_loss_list.append(log_loss(y_test, to_categorical(pred_test) ))
                    #labels=[0,1,2,3,4,5,6,7,8,9]))

        # Predict class of train set and store accuracy score
        pred_train = logreg_analyzer.predict(x_train)
        train_acc_list.append(accuracy(y_train, pred_train))
        train_loss_list.append(log_loss(y_train, to_categorical(pred_train)))


    return test_acc_list, train_acc_list, test_loss_list, train_loss_list, np.arange(1,epochs+1)


def plotter(xplot, yplot, plot_label_x, plot_label_y, plot_title, save_text, leg, yplot2 = None, title2=None):

    if yplot2 != None:  # Two subfigures
        figure = plt.figure(figsize=(10,8))
        plt.subplot(1,2,1)
        for y in yplot:
            plt.plot(xplot, y)
        plt.legend(leg)
        plt.title(plot_title)
        plt.xlabel(plot_label_x)
        plt.ylabel(plot_label_y)
        
        plt.subplot(1,2,2)
        for y2 in yplot2:
            plt.plot(xplot, y2)
        plt.legend(leg)
        plt.title(title2)
        plt.xlabel(plot_label_x)
        plt.ylabel(plot_label_y)
        
    else:               # Only one figure
        figure = plt.figure(figsize = (5,8))
        
        if isinstance(yplot, list): 
            for y in yplot:
                plt.plot(xplot, y)
        else:
            plt.plot(xplot, yplot)

        plt.xlabel(plot_label_x)
        plt.ylabel(plot_label_y)
        plt.title(plot_title)
        plt.legend(leg)

    plt.tight_layout()
    save_fig(save_text, folder=folder, extension='pdf')
    save_fig(save_text, folder=folder, extension='png')
    plt.show()



#print(train[0][0])

#print( train[1][0])

if "__main__" == __name__:
    x_train, y_train, x_test, y_test = mnist_data_for_logreg()

    logreg = LogReg(x_train, y_train, x_test, y_test)

    '''
    test_acc_list, train_acc_list, test_loss_list, \
            train_loss_list, epoch_interval = LogReg_analyze_mnist(x_train, y_train, x_test, y_test, 10, 30, 10000)

    # Make plot
    x_label = "epoch"
    y_label = "accuracy"
    title = "Accuracy vs. Epochs (Logistic Regression)"
    title2 = "Loss vs. Epochs"
    save_text = "epoch_vs_accuracy"
    leg = ["train", "test"]
    plotter(epoch_interval, [train_acc_list, test_acc_list], x_label, y_label, \
            title, save_text, leg, [train_loss_list, test_loss_list], title2)
    '''
