import random
import shutil
import os
import pandas as pd 
from matplotlib import pyplot as plt 
import cv2
import numpy as np 

# Select k random pictures from each subdirectory of source_path and copy them to destination_path
def select_random_pictures(source_path, destination_path, k):
    for root, dirs, files in os.walk(source_path):
        for directory in dirs:
            source_dir = os.path.join(root, directory)
            destination_dir = os.path.join(destination_path, os.path.relpath(source_dir, source_path))
            
            os.makedirs(destination_dir, exist_ok=True)
            
            picture_files = [file for file in os.listdir(source_dir) if file.endswith(('.jpg', '.jpeg', '.png'))]
            selected_files = random.sample(picture_files, min(k, len(picture_files)))
            
            for file in selected_files:
                source_file = os.path.join(source_dir, file)
                destination_file = os.path.join(destination_dir, file)
                shutil.copy2(source_file, destination_file)
             
# Plot a image read by opencv
def show_cv2_image(cv_img):
    image_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()
    
# Plot a image read by local path
def show_image_with_path(image_path):
    show_cv2_image(cv2.imread(image_path))  
    
# Add a column to the DataFrame
def add_column(df, col_name, default_value = np.NaN):
    if col_name not in df.columns:
        df[col_name] = default_value
    return df

# Get the full path of a folder under a specific path
def get_folders_under_path(path):
    return [os.path.join(path, x) for x in os.listdir(path) if os.path.isdir(os.path.join(path, x)) ]

# Define a function to compute the accuracy
def compute_accuracy(data, column_name, match_threshold = 70):
    c = data[column_name].astype(float)
    c[c >= match_threshold] = 1.0
    return (c.sum() / len(c))

# Plot the accuracy curve based on different confidence thresholds
def plot_accuracy_curve(data, column_name, min_threshold = 70, max_threshold = 100, step = 1):
    confidence_thresholds = np.arange(min_threshold, max_threshold, step)
    accuracies = [compute_accuracy(data, column_name, cft) for cft in confidence_thresholds]
    plt.plot(confidence_thresholds, accuracies)
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.ylim(0, 1)
    plt.show()
    
def plot_multiple_images(row, nums_in_row, images, figsize=(9, 4)):
    # Create a figure and subplots with a row of images
    fig, axs = plt.subplots(row, nums_in_row, figsize=figsize)
    # Adjust spacing between subplots
    fig.subplots_adjust(hspace=0, wspace=0)
    if axs.ndim == 1: axs = axs.reshape((1,-1))
    capabilities = row * nums_in_row
    min_one = min(capabilities, len(images))
    for i in range(min_one):
        image = images[i]
        row = i // nums_in_row
        column = i % nums_in_row
        axs[row][column].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axs[row][column].axis('off')
    # Adjust the spacing between subplots
    plt.tight_layout()
    plt.show()

