import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import os
import json


# def create_lenet5_model(input_shape=(64, 64, 1), num_classes=49, dropout_rate=0.3, l2_reg=0.001):
#     model = models.Sequential([
#         layers.Conv2D(6, (5, 5), activation='relu', input_shape=input_shape,
#                       kernel_regularizer=regularizers.l2(l2_reg)),  # L2 regularization
#         layers.AveragePooling2D(pool_size=(2, 2)),
#         layers.Dropout(dropout_rate),  # Dropout after convolution
#         layers.Conv2D(16, (5, 5), activation='relu', kernel_regularizer=regularizers.l2(l2_reg)),
#         layers.AveragePooling2D(pool_size=(2, 2)),
#         layers.Dropout(dropout_rate),  # Dropout after convolution
#         layers.Flatten(),
#         layers.Dense(120, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)),
#         layers.Dropout(dropout_rate),  # Dropout after dense layer
#         layers.Dense(84, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)),
#         layers.Dropout(dropout_rate),  # Dropout after dense layer
#         layers.Dense(num_classes, activation='softmax')  # Multi-class classification (softmax)
#     ])
#
#     # Adam optimizer with a learning rate of 0.0001 and gradient clipping
#     optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipvalue=1.0)
#
#     # Compile the model using categorical crossentropy (assuming one-hot encoded labels)
#     model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
#
#     return model


def create_lenet5_model(input_shape=(64, 64, 1), num_classes=49, dropout_rate=0.3, l2_reg=0.001):
    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(6, (5, 5), input_shape=input_shape,
                      kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),  # Batch Normalization
        layers.ReLU(),  # Activation after normalization
        layers.AveragePooling2D(pool_size=(2, 2)),
        layers.Dropout(dropout_rate),  # Dropout after convolution

        # Second convolutional block
        layers.Conv2D(16, (5, 5),
                      kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),  # Batch Normalization
        layers.ReLU(),  # Activation after normalization
        layers.AveragePooling2D(pool_size=(2, 2)),
        layers.Dropout(dropout_rate),  # Dropout after convolution

        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(120, kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),  # Batch Normalization
        layers.ReLU(),  # Activation after normalization
        layers.Dropout(dropout_rate),  # Dropout after dense layer

        layers.Dense(84, kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),  # Batch Normalization
        layers.ReLU(),  # Activation after normalization
        layers.Dropout(dropout_rate),  # Dropout after dense layer

        # Final output layer
        layers.Dense(num_classes, activation='softmax')  # Multi-class classification (softmax)
    ])

    # Compile the model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0001, clipvalue=1.0),  # Adam optimizer
        loss='categorical_crossentropy',  # Loss function for multi-class classification
        metrics=['accuracy']  # Evaluation metric
    )

    return model


def train_model_lenet5(train_dir, val_dir, version_path):
    # Create data generators for training and validation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,  # Normalize the image pixel values to [0, 1]
        rotation_range=20,  # Randomly rotate images by up to 20 degrees
        width_shift_range=0.2,  # Randomly shift the image horizontally by 20%
        height_shift_range=0.2,  # Randomly shift the image vertically by 20%
        shear_range=0.2,  # Shear the image randomly
        zoom_range=0.2,  # Zoom the image randomly
        horizontal_flip=True,  # Randomly flip images horizontally
        fill_mode='nearest'  # Fill the empty areas after transformations
    )
    val_datagen = ImageDataGenerator(rescale=1.0/255.0)

    # Flow data from directories
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(64, 64),
        color_mode='grayscale',
        batch_size=32,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(64, 64),
        color_mode='grayscale',
        batch_size=32,
        class_mode='categorical'
    )

    # Save class labels to a JSON file
    label_file = f"{version_path}/class_labels.json"
    class_labels = {v: k for k, v in train_generator.class_indices.items()}
    with open(label_file, 'w') as f:
        json.dump(class_labels, f, indent=4)

    # Build model
    model = create_lenet5_model()

    # Callbacks: Early stopping and model checkpoint to save the best model
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('lenet5_best_model.keras', monitor='val_loss', save_best_only=True)

    # Train the model with hyperparameter tuning and early stopping
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=50,
        validation_data=val_generator,
        validation_steps=val_generator.samples // val_generator.batch_size,
        callbacks=[early_stopping, model_checkpoint]
    )

    # Save training logs
    log_file = f"{version_path}/lenet_training_log,json"
    with open(log_file, 'w') as f:
        log_data = {
            "training_accuracy": history.history['accuracy'],
            "val_accuracy": history.history['val_accuracy'],
            "training_loss": history.history['loss'],
            "val_loss": history.history['val_loss'],
        }
        json.dump(log_data, f, indent=4)

    # Plot training and validation accuracy/loss
    plot_accuracy_loss(history, version_path)

    # Save the trained model
    model.save(f'{version_path}/lenet5_model.keras')

    return model


def plot_accuracy_loss(history, version_path):
    """Plot training and validation accuracy and loss."""
    plt.figure(figsize=(12, 6))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"{version_path}/lenet_training_accuracy.png")

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{version_path}/lenet_training_loss.png")

    plt.tight_layout()
    plt.show()
