import tensorflow as tf
import numpy as np
from tensorflow import keras
from LogReg import LogReg
from own_code import *

folder = "cats_and_dogs"


def accuracy(y, pred):
    y = y.ravel()
    pred = pred.ravel()
    return np.sum(y == pred) / len(y)



def cats_and_dogs_data_for_logreg():
    cats_and_dogs = tf.keras.datasets.cats_and_dogs     # Load dataset
    (x_train, y_train), (x_test, y_test) = cats_and_dogs.load_data()

    train_shape = x_train.shape
    test_shape  = x_test.shape

    x_train = x_train.ravel().reshape(train_shape[0], train_shape[1]*train_shape[2])/np.max(x_train)
    x_test =  x_test.ravel().reshape(test_shape[0], test_shape[1]*test_shape[2])/np.max(x_test)

    one_vector = np.ones((len(y_train),1))
    X_train1 = np.concatenate((one_vector, x_train), axis = 1)
    one_vector = np.ones((len(y_test),1))
    X_test1 = np.concatenate((one_vector, x_test), axis = 1)

    return X_train1, y_train, X_test1, y_test


def LogReg_analyze_cats_and_dogs(x_train, y_train, x_test, y_test, epochs, batch_size, n_image):
    n_train = n_image
    n_test = int(n_train/100)

    test_acc_list = []
    train_acc_list = []
    for i in range(1, epochs):
        logreg_analyzer = LogReg(x_train[0:n_train], y_train[0:n_train])
        logreg_analyzer.sgd(n_epochs= i, n_minibatches= batch_size)

        pred_test = logreg_analyzer.predict(x_test[0:n_test])
        test_acc_list.append(accuracy(y_test[0:n_test], pred_test))

        pred_train = logreg_analyzer.predict(x_train[0:n_train])
        train_acc_list.append(accuracy(y_train[0:n_train], pred_train))
        print(train_acc_list[-1])


    return test_acc_list, train_acc_list, np.arange(1,epochs)


def plotter(xplot, yplot, plot_label_x, plot_label_y, plot_title, save_text, leg, yplot2 = None):
    plt.title(plot_title)
    if yplot2 != None:
        plt.plot(xplot, yplot2)
    plt.plot(xplot, yplot)
    plt.xlabel(plot_label_x)
    plt.ylabel(plot_label_y)
    plt.title(plot_title)
    plt.legend(leg)
    plt.savefig("\Results\ " + save_text)
    plt.show()



#print(train[0][0])

#print( train[1][0])

if "__main__" == __name__:
    x_train, y_train, x_test, y_test = cats_and_dogs_data_for_logreg()

    test_acc_list, train_acc_list, epoch_interval = LogReg_analyze_cats_and_dogs(x_train, y_train, x_test, y_test, 10, 30, 10000)
    x_label = "epoch"
    y_label = "accuracy"
    title = "Epoch vs accuracy logistic regression"
    save_text = "epoch_vs_accuracy.pdf"
    leg = ["train", "test"]
    plotter(epoch_interval, train_acc_list, x_label, y_label, title, save_text, leg, test_acc_list)
