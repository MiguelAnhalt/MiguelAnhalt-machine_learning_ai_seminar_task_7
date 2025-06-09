# MIT License
#
# Copyright (c) 2017 François Chollet
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# TensorFlow and tf.keras
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from Task6_3_Load_image_and_process import main

# Configuration
USE_RGB = False  # Set to False for grayscale images
IMAGE_SIZE = (128, 128)  # Image dimensions (height, width)

# Load the custom dataset in grayscale mode
train_ds, validation_ds = main(IMAGE_SIZE[0], IMAGE_SIZE[1], 'grayscale')  # Explicitly request grayscale

# Get class names from the dataset
class_names = train_ds.class_names
print("Class names:", class_names)

# Convert the datasets to numpy arrays for compatibility with the rest of the code
train_images = []
train_labels = []
for images, labels in train_ds:
    # Ensure images are grayscale (1 channel)
    if images.shape[-1] == 3:  # If RGB, convert to grayscale
        images = tf.image.rgb_to_grayscale(images)
    train_images.extend(images.numpy())
    train_labels.extend(labels.numpy())
train_images = np.array(train_images)
train_labels = np.array(train_labels)

test_images = []
test_labels = []
for images, labels in validation_ds:
    # Ensure images are grayscale (1 channel)
    if images.shape[-1] == 3:  # If RGB, convert to grayscale
        images = tf.image.rgb_to_grayscale(images)
    test_images.extend(images.numpy())
    test_labels.extend(labels.numpy())
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Print dataset information
print("Training set shape:", train_images.shape)
print("Number of training samples:", len(train_labels))
print("Test set shape:", test_images.shape)
print("Number of test samples:", len(test_labels))


# Exploring the imported Fashion MIST dataset 
print(train_images.shape, len(train_labels), train_labels, test_images.shape, len(test_labels))
plt.figure()
plt.imshow(test_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# plot the first 25 images
plt.figure(figsize=(10,10))
for i in range(min(25, len(train_images))):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# Preprocess the data
# Scale these values to a range of 0 to 1 before feeding them to the neural network model
train_images = train_images / 255.0
test_images = test_images / 255.0

# and visualize
plt.figure(figsize=(10,10))
for i in range(min(25, len(train_images))):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# Building NN model - adjusted for grayscale images
input_shape = train_images.shape[1:]  # Get the shape of a single image (height, width, channels)
print("Input shape:", input_shape)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=input_shape),  # Input shape should be (128, 128, 1)
    tf.keras.layers.Flatten(),  # This will flatten to (128 * 128 * 1) = 16384
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(len(class_names))
])

# Print model summary to verify the architecture
model.summary()

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=60)

# Accuracy evaluation
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'\nTest accuracy: {test_acc * 100}%')

# Making predictions
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

# Functions for visualizing predictions
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap='gray')  # Changed to grayscale

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(len(class_names)))  # Changed to use actual number of classes
    plt.yticks([])
    thisplot = plt.bar(range(len(class_names)), predictions_array, color="#777777")  # Changed to use actual number of classes
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# Verifying predictions 
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
#num_images = num_rows*num_cols
num_images = min(num_rows * num_cols, len(predictions))
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

# Using the trained ANN model 
img = test_images[1]
img = (np.expand_dims(img,0))

predictions_single = probability_model.predict(img)

plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(len(class_names)), class_names, rotation=45)  # Changed to use actual number of classes
plt.show()
print("Predicted class:", class_names[np.argmax(predictions_single[0])])
