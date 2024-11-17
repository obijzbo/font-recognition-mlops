import os
import shutil
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.models import load_model
import glob
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import json
import numpy as np


def split_dataset(cleaned_data_path, train_data_path, test_data_path, split_ratio=0.8):
    """
    Splits the dataset into training and testing sets and stores them in the specified directories.
    """
    if not os.path.exists(cleaned_data_path):
        raise ValueError(f"Cleaned data path '{cleaned_data_path}' does not exist.")

    # Ensure the output directories exist
    os.makedirs(train_data_path, exist_ok=True)
    print(train_data_path)
    os.makedirs(test_data_path, exist_ok=True)
    print(test_data_path)

    # Process each label directory
    for label in os.listdir(cleaned_data_path):
        label_dir = os.path.join(cleaned_data_path, label)

        if not os.path.isdir(label_dir):
            continue

        # Get all image files for the current label
        images = [os.path.join(label_dir, img) for img in os.listdir(label_dir) if
                  img.endswith(('.png', '.jpg', '.jpeg'))]

        # Split into training and testing
        train_images, test_images = train_test_split(images, train_size=split_ratio, random_state=42)

        # Create subdirectories for the label in train and test paths
        train_label_dir = os.path.join(train_data_path, label)
        test_label_dir = os.path.join(test_data_path, label)
        os.makedirs(train_label_dir, exist_ok=True)
        os.makedirs(test_label_dir, exist_ok=True)

        # Move the files
        for train_image in train_images:
            shutil.copy(train_image, train_label_dir)
        for test_image in test_images:
            shutil.copy(test_image, test_label_dir)

    print(f"Data split completed: {split_ratio * 100}% training, {(1 - split_ratio) * 100}% testing.")


# Add Gaussian noise to the image
def add_noise(image, mean=0.0, stddev=0.1):
    noise = tf.random.normal(shape=tf.shape(image), mean=mean, stddev=stddev)
    noisy_image = tf.clip_by_value(image + noise, 0.0, 255.0)
    return noisy_image

# Apply Gaussian blur to the image
def apply_blur(image, kernel_size=3):
    # Using a blur filter as a workaround to simulate blur
    image = tf.image.resize(image, (64, 64))  # Ensure image size
    return tf.image.random_contrast(image, lower=0.9, upper=1.1)  # Simulating blur via contrast adjustment


def augment_training_data(train_data_path, batch_size=32, augmentations=None):
    """
    Augments the training data and stores the augmented images in the same training directory.

    Parameters:
    - train_data_path (str): Path to the directory containing the training data.
    - batch_size (int): The number of images to process at once.
    - augmentations (dict): Dictionary of augmentation parameters (e.g., rotation, flip).
    """
    if not os.path.exists(train_data_path):
        raise ValueError(f"Training data path '{train_data_path}' does not exist.")

    # Augmentation parameters, default if not passed
    if augmentations is None:
        augmentations = {
            'rotation': True,
            'noise': True,
            'blur': True,
            'flip': True,
            'brightness': True
        }

    # Load all the images (assuming images are .png or .jpg)
    image_paths = glob.glob(os.path.join(train_data_path, '**', '*.[jpg|png]'), recursive=True)

    image_paths = [str(path) for path in image_paths]

    # Prepare augmentation pipeline
    def augment_image(image):
        image = tf.image.resize(image, (64, 64))  # Ensure image is 64x64

        if augmentations['rotation']:
            image = tf.image.rot90(image)  # Rotates by 90 degrees

        if augmentations['flip']:
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)

        if augmentations['noise']:
            image = add_noise(image)

        if augmentations['blur']:
            image = apply_blur(image)

        if augmentations['brightness']:
            image = tf.image.random_brightness(image, max_delta=0.2)

        return image

    # Create a TensorFlow dataset from the image paths
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)

    # Load and augment images
    def load_and_augment(image_path):
        image = tf.io.read_file(tf.strings.as_string(image_path))
        image = tf.image.decode_png(image, channels=1)  # Grayscale image
        image = augment_image(image)
        return image, image_path  # Return both augmented image and original path for saving

    dataset = dataset.map(load_and_augment, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Save augmented images back to the training data path
    for idx, (images_batch, paths_batch) in enumerate(dataset):
        for i, (image, image_path) in enumerate(zip(images_batch, paths_batch)):
            # Extract the label folder from the original image path
            label_folder = os.path.basename(os.path.dirname(image_path))
            save_dir = os.path.join(train_data_path, label_folder)
            os.makedirs(save_dir, exist_ok=True)

            # Generate a new file name for the augmented image
            augmented_image_path = os.path.join(save_dir, f"augmented_{idx}_{i}.png")
            tf.io.write_file(augmented_image_path, tf.image.encode_png(image))

    print(f"Augmentation complete. Augmented images are saved in the original training directory: {train_data_path}")


def train_font_classification_model(data_path, model_path, log_path, initial_epochs=1, fine_tune_epochs=1):
    """
    Trains a font classification model using ResNet50 as the backbone.
    Implements TensorBoard for monitoring and fine-tunes the model for improved performance.

    Parameters:
    - data_path (str): Path to the directory containing the dataset.
    - initial_epochs (int): Number of epochs for initial training.
    - fine_tune_epochs (int): Number of epochs for fine-tuning.

    Returns:
    - model: The trained Keras model.
    - metrics: A dictionary containing the validation metrics (e.g., validation accuracy, loss).
    """
    # Load training dataset
    raw_train_ds = tf.keras.utils.image_dataset_from_directory(
        data_path,
        image_size=(64, 64),
        color_mode="grayscale",
        batch_size=32,
        validation_split=0.2,
        subset="training",
        seed=42
    )

    # Extract class names
    class_names = raw_train_ds.class_names
    num_classes = len(class_names)

    # Save class names to a JSON file
    class_names_file = "class_names.json"
    with open(class_names_file, "w") as f:
        json.dump(class_names, f)
    print(f"Class names saved to '{class_names_file}'")

    # Load validation dataset
    raw_val_ds = tf.keras.utils.image_dataset_from_directory(
        data_path,
        image_size=(64, 64),
        color_mode="grayscale",
        batch_size=32,
        validation_split=0.2,
        subset="validation",
        seed=42
    )

    # Normalize pixel values between 0 and 1
    normalization_layer = layers.Rescaling(1.0 / 255)

    train_ds = raw_train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = raw_val_ds.map(lambda x, y: (normalization_layer(x), y))

    # Prefetch for performance
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    # Model Definition: ResNet50
    base_model = tf.keras.applications.ResNet50(
        input_shape=(64, 64, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze base model initially

    model = models.Sequential([
        layers.InputLayer(input_shape=(64, 64, 1)),
        layers.Conv2D(3, (3, 3), padding='same', activation='relu'),  # Expand grayscale to 3 channels
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # TensorBoard Callback
    log_dir = os.path.join(log_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Early Stopping Callback
    early_stopping_cb = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    # Generate timestamp for checkpoint filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_filepath = f"{model_path}/model_checkpoint_{timestamp}.keras"

    # Checkpoint Callback
    checkpoint_cb = callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor='val_loss',
        save_best_only=True
    )

    # Initial Training
    print("Starting initial training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=initial_epochs,
        callbacks=[tensorboard_cb, early_stopping_cb, checkpoint_cb]
    )

    # Capture validation metrics after initial training
    val_loss_initial = history.history['val_loss']
    val_accuracy_initial = history.history['val_accuracy']

    # Fine-Tuning
    print("Starting fine-tuning...")
    base_model.trainable = True  # Unfreeze the base model for fine-tuning

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),  # Lower learning rate for fine-tuning
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    fine_tune_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=fine_tune_epochs,
        callbacks=[tensorboard_cb, early_stopping_cb]
    )

    # Capture validation metrics after fine-tuning
    val_loss_finetune = fine_tune_history.history['val_loss']
    val_accuracy_finetune = fine_tune_history.history['val_accuracy']

    # Save the final model
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_filename = f"{model_path}/font_classification_model_{timestamp}.keras"
    model.save(model_filename)
    print("Training complete. Model saved as 'font_classification_model.keras'")

    # Prepare metrics to return
    metrics = {
        'initial_training': {
            'val_loss': val_loss_initial[-1],
            'val_accuracy': val_accuracy_initial[-1]
        },
        'fine_tuning': {
            'val_loss': val_loss_finetune[-1],
            'val_accuracy': val_accuracy_finetune[-1]
        }
    }

    return model, metrics, class_names


def evaluate_model_on_test_data(model_path, test_data_path, log_path):
    """
    Evaluate a saved model on test data, save evaluation metrics, and plot a confusion matrix.

    Parameters:
    - model_path (str): Path to the saved model file.
    - test_data_path (str): Path to the directory containing the test dataset.
    - log_path (str): Path to save evaluation log file and confusion matrix.

    Returns:
    - evaluation_results (dict): Dictionary containing evaluation metrics and confusion matrix data.
    """
    # Load the model
    model = load_model(model_path)
    print(f"Model loaded from '{model_path}'.")

    # Load class names if available
    class_names_file = "class_names.json"
    if os.path.exists(class_names_file):
        with open(class_names_file, "r") as f:
            class_names = json.load(f)
        print(f"Class names loaded: {class_names}")
    else:
        class_names = None
        print(f"Class names file '{class_names_file}' not found.")

    # Load test dataset
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_data_path,
        image_size=(64, 64),
        color_mode="grayscale",
        batch_size=32,
        shuffle=False  # Do not shuffle for consistent evaluation
    )

    # Normalize pixel values between 0 and 1
    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))
    test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    # Get true labels and predictions
    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    y_pred = np.argmax(model.predict(test_ds), axis=-1)

    # Evaluate the model on the test dataset
    loss, accuracy = model.evaluate(test_ds)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")

    # Generate the confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    if class_names:
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
    else:
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)

    # Plot the confusion matrix
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    confusion_matrix_path = os.path.join(log_path, "confusion_matrix.png")
    plt.savefig(confusion_matrix_path)
    plt.close()
    print(f"Confusion matrix saved to '{confusion_matrix_path}'.")

    # Save evaluation metrics and confusion matrix data
    evaluation_results = {
        "test_loss": loss,
        "test_accuracy": accuracy,
        "confusion_matrix": conf_matrix.tolist()
    }
    if class_names:
        evaluation_results["class_names"] = class_names

    log_file = os.path.join(log_path, "evaluate_report.json")
    with open(log_file, "w") as log_file:
        json.dump(evaluation_results, log_file, indent=4)
    print(f"Evaluation results saved to '{log_file}'.")

    return evaluation_results
