# TensorFlow and tf.keras
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from Task6_3_Load_image_and_process import main

def apply_edge_detection(images):
    if isinstance(images, np.ndarray):
        images = tf.convert_to_tensor(images, dtype=tf.float32)
    if images.shape.rank == 3:
        images = tf.expand_dims(images, axis=-1)
    edges = tf.image.sobel_edges(images)
    magnitude = tf.sqrt(tf.reduce_sum(tf.square(edges), axis=-1))
    return magnitude.numpy()

# Configuration
USE_RGB = True  # Set to False for grayscale images
IMAGE_SIZE = (128, 128)  # Image dimensions (height, width)

# Load the custom dataset
color_mode = 'rgb' if USE_RGB else 'grayscale'
train_ds, validation_ds = main(IMAGE_SIZE[0], IMAGE_SIZE[1], color_mode)

# Get class names from the dataset
class_names = train_ds.class_names
print("Class names:", class_names)
print("Using", color_mode, "images")

# Convert the datasets to numpy arrays for compatibility with the rest of the code
train_images = []
train_labels = []
for images, labels in train_ds:
    train_images.extend(images.numpy())
    train_labels.extend(labels.numpy())
train_images = np.array(train_images)
train_labels = np.array(train_labels)

test_images = []
test_labels = []
for images, labels in validation_ds:
    test_images.extend(images.numpy())
    test_labels.extend(labels.numpy())
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Print dataset information
print("Training set shape:", train_images.shape)
print("Number of training samples:", len(train_labels))
print("Test set shape:", test_images.shape)
print("Number of test samples:", len(test_labels))

# Visualize the first image
plt.figure()
if USE_RGB:
    plt.imshow(test_images[0])
else:
    plt.imshow(test_images[0], cmap='gray')
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
    if USE_RGB:
        plt.imshow(train_images[i])
    else:
        plt.imshow(train_images[i], cmap='gray')
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# Preprocess the data
# Scale these values to a range of 0 to 1 before feeding them to the neural network model
train_images = train_images / 255.0
test_images = test_images / 255.0

# Building NN model - adjusted for image type
input_shape = train_images.shape[1:]  # Get the shape of a single image (height, width, channels)
print("Input shape:", input_shape)

# Calculate the number of input features based on the color mode
if USE_RGB:
    num_features = IMAGE_SIZE[0] * IMAGE_SIZE[1] * 3  # RGB: height * width * 3 channels
else:
    num_features = IMAGE_SIZE[0] * IMAGE_SIZE[1] * 1  # Grayscale: height * width * 1 channel
print("Number of input features:", num_features)

# Create the model with the correct input shape
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=input_shape),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
    tf.keras.layers.Dense(len(class_names))  # Output layer matches number of classes
])

# Print model summary to verify the architecture
model.summary()

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=40)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# Make predictions
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

# Functions for visualizing predictions
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    if USE_RGB:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')

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
    plt.xticks(range(len(class_names)))
    plt.yticks([])
    thisplot = plt.bar(range(len(class_names)), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# Plot a few test images and their predictions
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

# Test a single image
img = test_images[1]
img = (np.expand_dims(img,0))
predictions_single = probability_model.predict(img)
plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(len(class_names)), class_names, rotation=45)
plt.show()
print("Predicted class:", class_names[np.argmax(predictions_single[0])]) 