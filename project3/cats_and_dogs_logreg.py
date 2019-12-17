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


folder = "cats_and_dogs"

# load data
print("Loading data (cats and dogs)...")
x_train_cats = np.load('data/cats_train_images.npy')
y_train_cats = np.load('data/cats_train_categories.npy')
x_train_dogs = np.load('data/dogs_train_images.npy')
y_train_dogs = np.load('data/dogs_train_categories.npy')
x_test_cats = np.load('data/cats_test_images.npy')
y_test_cats = np.load('data/cats_test_categories.npy')
x_test_dogs = np.load('data/dogs_test_images.npy')
y_test_dogs = np.load('data/dogs_test_categories.npy')
print("Data loaded!\n")

x_train = np.concatenate((x_train_cats, x_train_dogs), axis=0)
x_test = np.concatenate((x_test_cats, x_test_dogs), axis=0)
y_train = np.concatenate((y_train_cats, y_train_dogs), axis=None)
y_test = np.concatenate((y_test_cats, y_test_dogs), axis=None)

# Scale data (min-max scaling)
x_train = (x_train - np.min(x_train))/(np.max(x_train) - np.min(x_train))
x_test = (x_test - np.min(x_test))/(np.max(x_test) - np.min(x_test))


# Shuffle the arrays so cats and dogs are not sorted
shuffled_indexes = np.random.permutation(len(x_train))
x_train = x_train[shuffled_indexes]
y_train = y_train[shuffled_indexes]
shuffled_indexes = np.random.permutation(len(x_test))
x_test = x_test[shuffled_indexes]
y_test = y_test[shuffled_indexes]

# data shape parameters
print(x_train.shape)
image_shape = x_train.shape[1:]
IMG_HEIGHT = image_shape[0]
IMG_WIDTH = image_shape[1]

print("\nINFO:")
print("The cats and dogs dataset")
print(f"Total number of observations: {len(y_train) + len(y_test)}")
print(f" - Of which {len(y_test)} are in the test set,")
print(f"   and {len(y_train)} are in the training set.")
print(f"Shape of the images: {IMG_HEIGHT} x {IMG_WIDTH}")
print(f"The shape of each image is {image_shape}")

# Plot a sample of the images
sample_indexes = np.random.choice(range(len(x_train)), size=5)
sample_images = x_train[sample_indexes]
#sample_images = np.random.choice(x_train, size=(5,IMG_HEIGHT, IMG_WIDTH))
plotImages(sample_images, folder=folder)
print(y_train[sample_indexes])

def LogReg(x_train, y_train, x_test, y_test):
    print("\nDoing logistic regression on the Cats vs Dogs data....")
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


def LogReg_analyze_cats_and_dogs(x_train, y_train, x_test, y_test, epochs, batch_size, n_image):
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



if "__main__" == __name__:
    #x_train, y_train, x_test, y_test = mnist_data_for_logreg()

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
