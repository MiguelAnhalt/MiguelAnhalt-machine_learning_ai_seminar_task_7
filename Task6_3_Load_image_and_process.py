# Task 6.3: Load image and process

# Import necessary libraries
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # (or “3” for even more suppression)
import tensorflow as tf             # For building and training models, and working with image datasets
import matplotlib.pyplot as plt     # For visualizing sample images
import os                           # For file and directory operations
import sys                          # For system-level operations like exiting the script


def set_data_path(img_height, img_width):
    """
    Sets the paths for training and test datasets and defines image dimensions.
    Returns:
        train_path: Path to training images
        test_path: Path to test images
        img_height: Target height of the images
        img_width: Target width of the images
    """
    # Get the directory where the current script is located
    dir_path = os.path.dirname(os.path.abspath(__file__))

    # Construct paths to training and test data
    train_path = os.path.join(dir_path, 'data', 'my_data','train')
    test_path = os.path.join(dir_path, 'data', 'my_data', 'test')

    # Check if training path exists, exit if it doesn't
    if not os.path.exists(train_path):
        print(f"Training data path does not exist: {train_path}")
        sys.exit(1)
    
    return train_path, test_path, img_height, img_width


def prepare_datasets(data_path_train, data_path_test, img_height, img_width, color_mode):
    """
    Loads and processes image data from directories into TensorFlow datasets.

    Args:
        data_path_train: Path to training data
        data_path_test: Path to test data
        img_height: Image height to resize to
        img_width: Image width to resize to

    Returns:
        ds_train: Training dataset
        ds_validation: Validation/test dataset
    """

    # Load training data and convert to tf.data.Dataset
    ds_train = tf.keras.preprocessing.image_dataset_from_directory(
        data_path_train,
        color_mode=color_mode,
        image_size=(img_height, img_width),  # Resize images
        shuffle=False,                       # Do not shuffle to keep image-label alignment
        seed=123,                            # Seed for reproducibility
        batch_size=32,                       # Batch size
    )

    # Load test/validation data
    ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
        data_path_test,
        color_mode='rgb',
        image_size=(img_height, img_width),  # Resize images
        shuffle=False,
        seed=123,
        batch_size=96,
    )
    
    return ds_train, ds_validation


def plot_sample_images(ds_train, ds_validation):
    """
    Plots a sample of images from training and validation datasets.

    Args:
        ds_train: Training dataset
        ds_validation: Validation dataset
    """

    # Get class names from training dataset
    if hasattr(ds_train, 'class_names'):
        class_names = ds_train.class_names
        print("Training class names:", class_names)
    else:
        raise AttributeError("ds_train does not have attribute 'class_names'.")

    # Plot sample training images
    plt.figure(figsize=(10, 10))
    for images, labels in ds_train:
        for i in range(min(18, images.shape[0])):  # Limit to first 18 images
            ax = plt.subplot(5, 4, i + 1)          # Create 5x4 grid
            plt.imshow(images[i].numpy().astype("uint8"))  # Convert tensor to image
            plt.title(class_names[labels[i]])              # Set title as class name
            plt.axis("off")
        break  # Only take the first batch
    plt.tight_layout()
    plt.show() 

    # Get class names from validation dataset
    if hasattr(ds_validation, 'class_names'):
        class_names = ds_validation.class_names
        print("Validation class names:", class_names)
    else:
        raise AttributeError("ds_validation does not have attribute 'class_names'.")

    # Plot sample validation images
    plt.figure(figsize=(10, 10))
    for images, labels in ds_validation:
        for i in range(min(9, images.shape[0])):  # Limit to first 9 images
            ax = plt.subplot(5, 3, i + 1)         # Create 5x3 grid
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
        break  # Only take the first batch
    plt.tight_layout()
    plt.show()  


# Main execution
def main(img_height_input, img_width_input, color_mode_input):    # Set paths to data and image size
    train_path, test_path, img_height, img_width = set_data_path(img_height_input, img_width_input)

    # Prepare TensorFlow datasets from image directories
    ds_train, ds_validation = prepare_datasets(train_path, test_path, img_height, img_width, color_mode_input)

    # Plot and visualize samples from datasets
    # plot_sample_images(ds_train, ds_validation)
    import numpy as np

    class_names = ds_train.class_names
    class_counts = np.zeros(len(class_names), dtype=int)

    for images, labels in ds_validation:
        for label in labels.numpy():
            class_counts[label] += 1

    for class_name, count in zip(class_names, class_counts):
        print(f"{class_name}: {count} images")

    return ds_train, ds_validation
