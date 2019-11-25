import os
import numpy as np
import matplotlib.pyplot as plt


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


