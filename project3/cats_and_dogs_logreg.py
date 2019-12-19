import tensorflow as tf
import numpy as np
from tensorflow import keras
#from LogReg import LogReg
from own_code import *
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.linear_model import LogisticRegression
from sklearn. metrics import log_loss
from skimage.color import rgb2gray
import time

from tensorflow.keras.losses import categorical_crossentropy
loss = categorical_crossentropy     # Loss function


folder = "cats_and_dogs_logreg"

# load data...
# if these .npy files are not in your directory, you must run the file
# cats_and_dogs_generate_data.py first. It will create the files.
# This is done because these .npy files were too big to upload to github
# therefore, we instead uploaded all the individual images, as well as the
# file that generates these .npy files from the original images.
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

# Reshape the data, and convert to grayscale
x_train = rgb2gray(x_train)
x_test = rgb2gray(x_test)
print("Converting to grayscale images...")
print(x_train.shape)


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
plotImages(sample_images, folder=folder, rgb=False)
print(y_train[sample_indexes])

# Flatten the images
x_train = x_train.ravel().reshape(x_train.shape[0], IMG_HEIGHT*IMG_WIDTH)
x_test = x_test.ravel().reshape(x_test.shape[0], IMG_HEIGHT*IMG_WIDTH)
print(x_train.shape)

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
    plotMisclassifiedImages(y_test, x_test.reshape(imgShape[0],IMG_HEIGHT,IMG_WIDTH), pred_test,
            figurename="sample_misclassified_images", folder=folder, rgb=False)

    return logreg


if "__main__" == __name__:
    #x_train, y_train, x_test, y_test = mnist_data_for_logreg()

    logreg = LogReg(x_train, y_train, x_test, y_test)
