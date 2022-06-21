from fashion import load, network
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
# from keras.utils import plot_model

import datetime

test = load.load_cifar_10()
# print(test)
(train_images, train_labels), (test_images, test_labels) = test
# plt.figure(figsize=(10, 5))
# plt.imshow(train_images[0])
# plt.colormaps()
# plt.show()
train_images = train_images / 255.
test_images = test_images / 255
# model.add(Conv2D(32, (3, 3), activation='relu',
#           kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
# model.add(MaxPooling2D((2, 2)))
# model.add(Flatten())
model = tf.keras.applications.EfficientNetB1(
    # include_top=False,
    weights=None,
    input_tensor=None,
    input_shape=(32, 32, 3),
    pooling=None,
    classes=10,
    classifier_activation='softmax',
    # include_preprocessing=True
)
testModel = keras.Model(inputs=model.input,
                        outputs=model.get_layer('top_activation').output)
testModel.summary()
# model=network.getInterOutModel()
# model = keras.Sequential([
#     keras.layers.Conv2D(32, (3, 3), activation='relu',
#                         kernel_initializer='he_uniform', input_shape=(28, 28, 1)),
#     keras.layers.MaxPooling2D((2, 2)),
#     keras.layers.Flatten(),
#     keras.layers.Dense(256, activation=tf.nn.relu),
#     keras.layers.Dense(10, activation=tf.nn.softmax)
# ])
keras.utils.plot_model(
    testModel, to_file='test_model_top.png', show_shapes=True)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1)
earlyStoppingCb = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', patience=5)
model.fit(train_images, train_labels, epochs=50, callbacks=[
    earlyStoppingCb, tensorboard_callback], validation_split=0.2)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Model Accuracy: {test_acc * 100}%")
