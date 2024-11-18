import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import json
import pandas as pd

def evaluate_model(model_name, label_file, test_dir, current_version_dir):
    # Load the trained model
    model_path = f"{current_version_dir}/{model_name}"
    model = tf.keras.models.load_model(model_path)

    # Load class labels from the provided JSON file
    with open(f"{current_version_dir}/{label_file}", 'r') as f:
        class_labels = json.load(f)

    # Create a test data generator
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    # Flow data from the test directory
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(64, 64),
        color_mode='grayscale',
        batch_size=32,
        class_mode='categorical',  # Use categorical for multi-class classification
        shuffle=False  # Important: Do not shuffle for evaluation
    )

    # Calculate the total number of steps required (including the last incomplete batch)
    total_steps = (test_generator.samples + test_generator.batch_size - 1) // test_generator.batch_size

    # Predict labels for the test data
    predictions = model.predict(test_generator, steps=total_steps, verbose=1)

    # For multi-class, the output of predictions is a probability distribution over the classes.
    # We need to convert these probabilities into class labels (the class with the maximum probability).
    predicted_labels = predictions.argmax(axis=-1)

    # Get the true labels from the test data generator
    true_labels = test_generator.classes  # Ground truth labels from the generator

    # Ensure that true labels and predicted labels have the same length
    if len(true_labels) != len(predicted_labels):
        raise ValueError(f"Number of true labels ({len(true_labels)}) does not match number of predicted labels ({len(predicted_labels)})")

    accuracy = accuracy_score(true_labels, predicted_labels)
    accuracy_percentage = accuracy * 100

    # Generate classification report
    report = classification_report(true_labels, predicted_labels, target_names=list(class_labels.values()))

    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    confusion_df = pd.DataFrame(conf_matrix, index=list(class_labels.values()), columns=list(class_labels.values()))

    # Save the classification report to a file
    report_file = f"{current_version_dir}/{model_name}_performance_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)

    cm_file = f"{current_version_dir}/{model_name}_confusion_matrix.csv"
    confusion_df.to_csv(cm_file)

    return accuracy_percentage
