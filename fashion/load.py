import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np


def load_fashion_mnist():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images,
                                   test_labels) = fashion_mnist.load_data()
    return (train_images, train_labels), (test_images, test_labels)
