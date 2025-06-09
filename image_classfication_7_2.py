# MIT License (c) 2017 François Chollet

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from Task6_3_Load_image_and_process import main

# Configuration
USE_RGB = False
IMAGE_SIZE = (128, 128)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])

train_ds, validation_ds = main(IMAGE_SIZE[0], IMAGE_SIZE[1], 'grayscale')
class_names = train_ds.class_names
print("Class names:", class_names)

# Augment training data
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

# Convert datasets to numpy arrays
train_images, train_labels = [], []
for images, labels in train_ds:
    if images.shape[-1] == 3:
        images = tf.image.rgb_to_grayscale(images)
    train_images.extend(images.numpy())
    train_labels.extend(labels.numpy())
train_images, train_labels = np.array(train_images), np.array(train_labels)

test_images, test_labels = [], []
for images, labels in validation_ds:
    if images.shape[-1] == 3:
        images = tf.image.rgb_to_grayscale(images)
    test_images.extend(images.numpy())
    test_labels.extend(labels.numpy())
test_images, test_labels = np.array(test_images), np.array(test_labels)

print("Training set shape:", train_images.shape)
print("Test set shape:", test_images.shape)

# Normalize images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Model definition using CNN
input_shape = train_images.shape[1:]
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=input_shape),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(class_names))
])
model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=60,
                    validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"\nTest accuracy: {test_acc * 100:.2f}%")

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')
plt.show()

# Predictions and evaluation
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
y_pred = np.argmax(predictions, axis=1)

# Confusion matrix
cm = confusion_matrix(test_labels, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Classification report
print(classification_report(test_labels, y_pred, target_names=class_names))

# Save model
model.save("grayscale_model.h5")
print("Model saved as grayscale_model.h5")

# Example prediction plot
def plot_value_array(predictions_array, true_label):
    true_label = true_label
    plt.grid(False)
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.yticks([])
    thisplot = plt.bar(range(len(class_names)), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

sample_index = 1
img = np.expand_dims(test_images[sample_index], 0)
predictions_single = probability_model.predict(img)
plot_value_array(predictions_single[0], test_labels[sample_index])
plt.title(f"Predicted: {class_names[np.argmax(predictions_single[0])]}")
plt.show()
