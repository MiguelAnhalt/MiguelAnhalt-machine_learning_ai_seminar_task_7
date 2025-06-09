import os
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
from Task6_3_Load_image_and_process import main

# ---------------------------------------------
# EDGE DETECTION FUNCTION
# ---------------------------------------------
def apply_edge_detection(images):
    if isinstance(images, np.ndarray):
        images = tf.convert_to_tensor(images, dtype=tf.float32)
    if images.shape.rank == 3:
        images = tf.expand_dims(images, axis=-1)
    edges = tf.image.sobel_edges(images)
    magnitude = tf.sqrt(tf.reduce_sum(tf.square(edges), axis=-1))
    return magnitude.numpy()

# ---------------------------------------------
# LOAD & PREPROCESS DATA
# ---------------------------------------------
def load_and_preprocess_data():
    IMAGE_SIZE = (128, 128)
    train_ds, validation_ds = main(IMAGE_SIZE[0], IMAGE_SIZE[1], 'rgb')
    class_names = train_ds.class_names

    def dataset_to_numpy(ds):
        images, labels = [], []
        for batch_images, batch_labels in ds:
            images.extend(batch_images.numpy())
            labels.extend(batch_labels.numpy())
        return np.array(images), np.array(labels)

    train_images, train_labels = dataset_to_numpy(train_ds)
    test_images, test_labels = dataset_to_numpy(validation_ds)

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    train_images = apply_edge_detection(train_images)
    test_images = apply_edge_detection(test_images)

    # Reshape edge-detected images to (batch, 128, 128, 1)
    if train_images.ndim == 3:
        train_images = np.expand_dims(train_images, -1)
        test_images = np.expand_dims(test_images, -1)

    return train_images, train_labels, test_images, test_labels, class_names

# ---------------------------------------------
# BUILD MODEL
# ---------------------------------------------
'''def build_transfer_model(input_shape, num_classes):
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False

    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(3, (3, 3), padding="same", activation="relu"),  # Expand grayscale to 3 channels
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes)
    ])
    return model'''

def build_custom_cnn(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes)
    ])
    return model


# ---------------------------------------------
# COMPILE & TRAIN
# ---------------------------------------------
def model_compile(model):
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

def train_model(model, train_images, train_labels, val_images, val_labels, epochs=10):
    #callback = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    history = model.fit(train_images, train_labels,
                        validation_data=(val_images, val_labels),
                        epochs=epochs, batch_size=96)
                        #callbacks=[callback])
    return history

# ---------------------------------------------
# EVALUATION
# ---------------------------------------------
def accuracy_evaluation(model, test_images, test_labels):
    loss, acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f"\nTest Accuracy: {acc * 100:.2f}%")
    return loss, acc

# ---------------------------------------------
# PREDICTION VISUALIZATION
# ---------------------------------------------
def predict(model, test_images):
    prob_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    return prob_model.predict(test_images)

def plot_image(i, predictions_array, true_label, img, class_names):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img.squeeze(), cmap='gray')
    pred_label = np.argmax(predictions_array)
    color = 'blue' if pred_label == true_label else 'red'
    plt.xlabel(f"{class_names[pred_label]} {100*np.max(predictions_array):.0f}% ({class_names[true_label]})", color=color)

def plot_value_array(i, predictions_array, true_label, class_names):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(len(class_names)))
    plt.yticks([])
    bar = plt.bar(range(len(class_names)), predictions_array, color="#777777")
    plt.ylim([0, 1])
    pred_label = np.argmax(predictions_array)
    bar[pred_label].set_color('red')
    bar[true_label].set_color('blue')

# ---------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------
def main_execution():
    train_images, train_labels, test_images, test_labels, class_names = load_and_preprocess_data()
    #model = build_transfer_model(input_shape=train_images.shape[1:], num_classes=len(class_names))
    model = build_custom_cnn(input_shape=train_images.shape[1:], num_classes=len(class_names))
    model = model_compile(model)
    train_model(model, train_images, train_labels, test_images, test_labels, epochs=70)
    accuracy_evaluation(model, test_images, test_labels)

    predictions = predict(model, test_images)

    # Visualize
    num_rows, num_cols = 5, 3
    num_images = min(num_rows * num_cols, len(predictions))
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i, predictions[i], test_labels, test_images, class_names)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(i, predictions[i], test_labels, class_names)
    plt.tight_layout()
    plt.show()

# Run
if __name__ == '__main__':
    main_execution()
