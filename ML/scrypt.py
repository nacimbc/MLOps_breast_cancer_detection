#Import necessary libraries
from tensorflow import keras
from keras import layers
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow_datasets as tfds

import os, shutil, pathlib, glob
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import pandas as pd
import random
import kagglehub

SEED = 4747
random.seed(SEED)

print("everything is okay")


# Download latest version
path = kagglehub.dataset_download("paultimothymooney/breast-histopathology-images")

print(".", path)