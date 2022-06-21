import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np


def getInterOutModel(model, layerName):
    return keras.Model(inputs=model.input,
                       outputs=model.get_layer(layerName).output)
