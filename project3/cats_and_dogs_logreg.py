
import numpy as np
import matplotlib.pyplot as plt
import os
from own_code import *
import time

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from skimage.color import rgb2gray
from sklearn.linear_model import LogisticRegression


def cats_and_dogs_data_for_logreg():
    _URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
    path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
    # This is the path to the data
    PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

    # Path to the training and the testing data
    train_dir = os.path.join(PATH, 'train')
    validation_dir = os.path.join(PATH, 'validation')
    folder = "cats_and_dogs"

    # Path to the training/test data divided into cats and dogs
    train_cats_dir = os.path.join(train_dir, 'cats')  # directory with our training cat pictures
    train_dogs_dir = os.path.join(train_dir, 'dogs')  # directory with our training dog pictures
    validation_cats_dir = os.path.join(validation_dir, 'cats')  # directory with our validation cat pictures
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # directory with our validation dog pictures

    # Collect info about the dataset
    num_cats_tr = len(os.listdir(train_cats_dir))
    num_dogs_tr = len(os.listdir(train_dogs_dir))
    num_cats_val = len(os.listdir(validation_cats_dir))
    num_dogs_val = len(os.listdir(validation_dogs_dir))
    total_train = num_cats_tr + num_dogs_tr
    total_val = num_cats_val + num_dogs_val

    # Print info
    print("DATASET INFO:")
    print('Total training cat images:', num_cats_tr)
    print('Total training dog images:', num_dogs_tr)
    print('Total validation cat images:', num_cats_val)
    print('Total validation dog images:', num_dogs_val)
    print("--")
    print("Total training images:", total_train)
    print("Total validation images:", total_val)
    print("_______________________")

    IMG_HEIGHT = 150
    IMG_WIDTH = 150

    # Scale (normalize) data
    train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
    validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

    # Create generators that generates the images
    train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           color_mode = color_mode,
                                                           class_mode='binary')

    val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                            directory=validation_dir,
                                                            target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                            color_mode = color_mode,
                                                            class_mode='binary')

    # Plot 5 randomly selected images from the training set
    #sample_training_images, _ = next(train_data_gen) 
    #plotImages(sample_training_images[:5], folder="cats_and_dogs_logreg")


def LogReg():
    print("Doing logistic regression on the cats and dogs data...")
    start_time = time.time()

    # Fit logreg class
    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)    # !!!



    # Print resutls
    print("Results:")
    print(f"\n     |  Train    |  Test")
    print("------------------------------------")
    print(f"Acc  |{train_acc:10.4f} |{test_acc:10.4f}")
    print(f"Loss |{train_loss:10.4f} |{test_loss:10.4f}\n")


    

