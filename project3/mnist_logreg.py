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
    
    return x_train, y_train, x_test, y_test

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




if "__main__" == __name__:
    x_train, y_train, x_test, y_test = mnist_data_for_logreg()

    logreg = LogReg(x_train, y_train, x_test, y_test)

