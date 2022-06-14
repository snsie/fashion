from fashion import load
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

test = load.load_fashion_mnist()
print(test)
(train_images, train_labels), (test_images, test_labels) = test
plt.figure(figsize=(10, 5))
plt.imshow(train_images[0])
plt.colormaps()
plt.show()
train_images = train_images / 255.
test_images = test_images / 255
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
keras.utils.plot_model(model, to_file='model.png', show_shapes=True)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(train_images, train_labels, epochs=5)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Model Accuracy: {test_acc * 100}%")
