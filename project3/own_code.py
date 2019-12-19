import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import tensorflow as tf
import seaborn
import pandas as pd


# Where to save the figures and data files
PROJECT_ROOT_DIR = "Results"
FIGURE_ID = "Results/FigureFiles"
#DATA_ID = "DataFiles/"

# Make these folders if they do not exist
if not os.path.exists(PROJECT_ROOT_DIR):
    os.mkdir(PROJECT_ROOT_DIR)
if not os.path.exists(FIGURE_ID):
    os.makedirs(FIGURE_ID)
#if not os.path.exists(DATA_ID):
#    os.makedirs(DATA_ID)


def image_path(fig_id, folder=""):
    """ Function to create path for new figure """
    if not os.path.exists(os.path.join(FIGURE_ID, folder)):
        os.makedirs(os.path.join(FIGURE_ID, folder))
    return os.path.join(FIGURE_ID, folder, fig_id)

def save_fig(fig_id, folder="", extension='pdf'):
    """ Function for saving figure in correct folder """
    plt.savefig(image_path(fig_id, folder=folder) + '.' + extension, format=extension)

#def data_path(dat_id):
#    return os.path.join(DATA_ID, dat_id)


# Function for plotting a selected number of images from the training
# set in one figure
def plotImages(images_arr, folder= "", rgb=True, show=False):
    fig, axes = plt.subplots(1, len(images_arr), figsize=(20,4))
    axes = axes.flatten()
    grayscale = False
    for img, ax in zip(images_arr, axes):
        if img.shape[-1] == 1:                  # If image is grayscale
            print("image is grayscale")
            grayscale=True
            IMG_HEIGHT = images_arr[0].shape[0]
            IMG_WIDTH = images_arr[0].shape[1]
            img = img.reshape(IMG_HEIGHT, IMG_WIDTH)    # Reshape
            ax.imshow(img, cmap='gray')
        elif rgb == False:
            ax.imshow(img, cmap = 'gray')
        else:
            ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()

    if folder == "mnist":
        figure_name = "sample_images_MNIST"
    elif folder == "cats_and_dogs":
        figure_name = "sample_images_CD"
    else:
        figure_name = "sample_images"

    if grayscale:       # If grayscale
        save_fig(figure_name + "_(grayscale)", folder=folder, extension='pdf')
        save_fig(figure_name + "_(grayscale)", folder=folder, extension='png')
    else:
        save_fig(figure_name, folder=folder, extension='pdf')
        save_fig(figure_name, folder=folder, extension='png')
    if show == True:
        plt.show()
    else:
        plt.close()


def confusionMatrix(ypred, ytrue, figurename=None, folder='', show=False):
    categories = np.arange(np.max(ytrue) + 1)
    n_categories = len(categories)      # Find the number of categories    
    con_matr = tf.math.confusion_matrix(labels=ytrue, predictions=ypred).numpy()
    con_matr_normalized = np.around(con_matr.astype('float')/con_matr.sum(axis=1)[:, np.newaxis], decimals=2)  

    # save confusion matrix in dataframe and print
    df = pd.DataFrame(con_matr_normalized, index = categories, columns = categories) 
    print(f"\nConfusion matrix: \n{df}")

    # plot
    figure = plt.figure()
    seaborn.heatmap(con_matr_normalized, annot=True, cmap=plt.cm.Purples)

    # fix for matplotlib (3.1.1) bug that cuts off top/bottom of seaborn viz
    b, t = plt.ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t) # update the ylim(bottom, top) values

    plt.tight_layout()
    plt.ylabel("True category")
    plt.xlabel("Predicted category")
    
    if figurename:
        save_fig(figurename, folder=folder, extension='pdf')
        save_fig(figurename, folder=folder, extension='png')

    if show:
        plt.show()
    else:
        plt.close()

    return df

def accuracy(y, pred):
    """ Function for calculating classification accuracy """
    y = y.ravel()
    pred = pred.ravel()
    return np.sum(y == pred) / len(y)


def plotMisclassifiedImages(y_true, images, y_pred, figurename=None, folder="", rgb=True, show=True):
    misclassified_inds = []     # The indexes of the misclassified images
    index = 0
    for label, pred in zip(y_true, y_pred):
        if label != pred:
            misclassified_inds.append(index)
        index += 1
    misclassified_imgs = [images[i] for i in misclassified_inds]
    images_arr = misclassified_imgs

    # Show the misclassified images in figure
    n_images = len(misclassified_inds)
    if n_images < 10:       # Plot 5 or less of the images
        n_images = min(5, n_images)

        fig, axes = plt.subplots(1, n_images, figsize=(4*n_images,4))
        axes = axes.flatten()
        grayscale = False
        for img, ax in zip(images_arr, axes):
            if img.shape[-1] == 1:                  # If image is grayscale
                grayscale=True
                IMG_HEIGHT = images_arr[0].shape[0]
                IMG_WIDTH = images_arr[0].shape[1]
                img = img.reshape(IMG_HEIGHT, IMG_WIDTH)    # Reshape
                ax.imshow(img, cmap='gray')
            elif rgb == False:
                ax.imshow(img, cmap = 'gray')
            else:
                ax.imshow(img)
            ax.set_title(f"Predicted: {y_pred[misclassified_inds[i]]}, Actual: {y_true[misclassified_inds[i]]}",
                            fontweight='bold', size=20)
            i += 1
            ax.axis('off')
            plt.tight_layout()
        plt.tight_layout()

    elif n_images >= 10:    # Plot 10 of the images
        fig, axes = plt.subplots(2, 5, figsize = (20, 8))
        grayscale = False
        axes = axes.flatten()
        i = 0       # counter
        for img, ax in zip(images_arr, axes):
            if img.shape[-1] == 1:                  # If image is grayscale
                grayscale=True
                IMG_HEIGHT = images_arr[0].shape[0]
                IMG_WIDTH = images_arr[0].shape[1]
                img = img.reshape(IMG_HEIGHT, IMG_WIDTH)    # Reshape
                ax.imshow(img, cmap='gray')
            elif rgb == False:
                ax.imshow(img, cmap = 'gray')
            else:
                ax.imshow(img)

            ax.set_title(f"Predicted: {y_pred[misclassified_inds[i]]}, Actual: {y_true[misclassified_inds[i]]}",
                            fontweight='bold', size=20)
            i += 1
            ax.axis('off')
            plt.tight_layout()
        plt.tight_layout()

   
    # Save figure if name is given
    if figurename != None:
        figurename = figurename + "_" + folder
        save_fig(figurename, folder=folder, extension='pdf')
        save_fig(figurename, folder=folder, extension='png')

    if show:
        plt.show()
    else:
        plt.close()

    return misclassified_imgs


