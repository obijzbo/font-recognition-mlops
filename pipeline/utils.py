import os
import shutil
from sklearn.model_selection import train_test_split
import json


def create_directory_if_not_exists(directory_path):
    """
    Creates a directory if it does not already exist.

    Parameters:
    directory_path (str): The path of the directory to create.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")


def track_version(directory_path):
    try:
        child_dir = [name for name in os.listdir(directory_path) if
                             os.path.isdir(os.path.join(directory_path, name))]
        version_number = len(child_dir) + 1
        return version_number
    except FileNotFoundError:
        create_directory_if_not_exists(directory_path)
        return 1


def remove_directory_if_exists(directory_path):
    """
    Removes a directory and all of its contents if it exists.

    Parameters:
    directory_path (str): The path of the directory to remove.
    """
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        shutil.rmtree(directory_path)
        print(f"Directory '{directory_path}' and all its contents have been removed.")
    else:
        print(f"Directory '{directory_path}' does not exist.")


def split_dataset(cleaned_data_path, train_data_path, test_data_path, val_data_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Splits the dataset into training, validation, and testing sets and stores them in the specified directories.

    Args:
        cleaned_data_path (str): Path to the cleaned data.
        train_data_path (str): Path to store the training data.
        test_data_path (str): Path to store the testing data.
        val_data_path (str): Path to store the validation data.
        train_ratio (float): Proportion of data to use for training.
        val_ratio (float): Proportion of data to use for validation.
        test_ratio (float): Proportion of data to use for testing.
    """
    if not os.path.exists(cleaned_data_path):
        raise ValueError(f"Cleaned data path '{cleaned_data_path}' does not exist.")

    # Ensure the output directories exist
    os.makedirs(train_data_path, exist_ok=True)
    os.makedirs(test_data_path, exist_ok=True)
    os.makedirs(val_data_path, exist_ok=True)

    # Process each label directory
    for label in os.listdir(cleaned_data_path):
        label_dir = os.path.join(cleaned_data_path, label)

        if not os.path.isdir(label_dir):
            continue

        # Get all image files for the current label
        images = [os.path.join(label_dir, img) for img in os.listdir(label_dir) if
                  img.endswith(('.png', '.jpg', '.jpeg'))]

        # Split into training, validation, and testing
        train_images, temp_images = train_test_split(images, train_size=train_ratio, random_state=42)
        val_split_ratio = val_ratio / (val_ratio + test_ratio)
        val_images, test_images = train_test_split(temp_images, train_size=val_split_ratio, random_state=42)

        # Create subdirectories for the label in train, val, and test paths
        train_label_dir = os.path.join(train_data_path, label)
        val_label_dir = os.path.join(val_data_path, label)
        test_label_dir = os.path.join(test_data_path, label)
        os.makedirs(train_label_dir, exist_ok=True)
        os.makedirs(val_label_dir, exist_ok=True)
        os.makedirs(test_label_dir, exist_ok=True)

        # Move the files
        for train_image in train_images:
            shutil.copy(train_image, train_label_dir)
        for val_image in val_images:
            shutil.copy(val_image, val_label_dir)
        for test_image in test_images:
            shutil.copy(test_image, test_label_dir)

    print(f"Data split completed: {train_ratio * 100}% training, {val_ratio * 100}% validation, {test_ratio * 100}% testing.")


def model_performance_track(model_performance_log_file, model_file_path, model_accuracy, version):
    # Check if the file exists
    if os.path.exists(model_performance_log_file):
        # Read the existing data from the file
        with open(model_performance_log_file, 'r') as f:
            data = json.load(f)

        # Get the existing accuracy from the file
        existing_accuracy = data.get('model_accuracy', -1)

        # Compare the existing accuracy with the new accuracy
        if model_accuracy > existing_accuracy:
            # Update the file only if the new accuracy is better
            data['model_file_path'] = model_file_path
            data['model_accuracy'] = model_accuracy
            data['version'] = version

            # Write the updated data back to the file
            with open(model_performance_log_file, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"File updated with new accuracy: {model_accuracy}%")
        else:
            print(
                f"Accuracy {model_accuracy}% is not higher than the existing accuracy {existing_accuracy}%. No update made.")
    else:
        # If the file doesn't exist, create a new one with the given data
        data = {
            'model_file_path': model_file_path,
            'model_accuracy': model_accuracy,
            'version': version
        }
        with open(model_performance_log_file, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"New file created with accuracy: {model_accuracy}%")
