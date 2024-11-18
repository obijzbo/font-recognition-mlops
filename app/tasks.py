from celery import Celery
import cv2
from PIL import Image
import tensorflow as tf
import numpy as np
import io
import json
import os


# Initialize Celery
celery_app = Celery(
    "tasks",
    broker="redis://redis:6379/0",  # Use Redis as the message broker
    backend="redis://redis:6379/0"  # Use Redis as the result backend
)

# Load TensorFlow model
MODEL_NAME = os.getenv("MODEL_NAME", "/app/lenet5_model.keras")
model = tf.keras.models.load_model(MODEL_NAME)


def load_class_labels():
    with open('class_labels.json', 'r') as f:
        return json.load(f)


# Define a function to get the human-readable label
class_labels = load_class_labels()


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image at {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Invert colors
    inverted = cv2.bitwise_not(gray)

    # Apply thresholding for binarization
    _, thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cv2.imwrite(image_path, thresh)


def segment_fonts(output_dir, image_path):
    """Segment words from images and save them in their labeled directories."""

    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        segmeted_path = []
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            segmented_word = binary_image[y:y + h, x:x + w]

            word_file_name = f"{os.path.splitext(image_path)[0]}_font_{i}.png"
            word_output_path = os.path.join(output_dir, word_file_name)
            cv2.imwrite(word_output_path, segmented_word)
            segmeted_path.append(word_output_path)
        return segmeted_path
    except Exception as e:
        print(f"Error processing {image_path}: {e}")


def resize_with_padding(img_path, output_dir, target_size=(128, 128)):
            img = Image.open(img_path)

            # Ensure the image retains its original mode
            mode = img.mode
            if mode not in ["L", "1"]:  # Convert to grayscale if not already
                img = img.convert("L")

            # Get original dimensions
            original_width, original_height = img.size
            target_width, target_height = target_size

            # Safeguard against division by zero
            if original_height == 0 or original_width == 0:
                print(f"Skipping image {img_path} with invalid dimensions: {img.size}")

            # Calculate the aspect ratio of the original image
            original_aspect = original_width / original_height
            target_aspect = target_width / target_height

            # Resize image based on aspect ratio
            if original_aspect > target_aspect:
                # Image is wider; fit to target width and adjust height
                new_width = target_width
                new_height = int(target_width / original_aspect)
            else:
                # Image is taller; fit to target height and adjust width
                new_height = target_height
                new_width = int(target_height * original_aspect)

            # Check if the new width or height is valid
            if new_width <= 0 or new_height <= 0:
                print(f"Skipping image {img_path} with invalid resized dimensions: ({new_width}, {new_height})")

            # Resize to the new dimensions
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Create a new canvas with the target size and black background
            background_color = 0 if mode == "L" else 0  # Black background for both grayscale and binary
            padded_img = Image.new(mode, target_size, background_color)

            # Calculate the position to paste the resized image to center it
            paste_position = (
                (target_width - new_width) // 2,
                (target_height - new_height) // 2
            )
            padded_img.paste(img, paste_position)

            # Save the padded image
            padded_img.save(os.path.join(output_dir, img_path))
            return os.path.join(output_dir, img_path)


@celery_app.task
def preprocess_and_predict(image_bytes):
    """
    Celery task to preprocess, segment, resize, and predict.
    """
    try:
        # Load the image
        input_image = Image.open(io.BytesIO(image_bytes))
        input_image = input_image.convert("L")

        # Save to a temporary file
        temp_image_path = "/tmp/temp_input_image.jpg"
        input_image.save(temp_image_path)

        # Step 1: Preprocess the image
        preprocessed_image_path = preprocess_image(temp_image_path)
        print("Image preprocessed!!!")

        # Step 2: Segment the image
        segmented_image_paths = segment_fonts(os.path.dirname(preprocessed_image_path), preprocessed_image_path)
        print(f"Image segmented into {len(segmented_image_paths)} words!!!")

        # Step 3: Resize and pad each segment
        resized_image_paths = [resize_with_padding(img_path, os.path.dirname(preprocessed_image_path), (64, 64)) for
                               img_path in segmented_image_paths]
        print("Images resized and padded!!!")

        # Step 4: Load images for prediction and prepare them
        cleaned_images = []
        for img_path in resized_image_paths:
            img = Image.open(img_path)
            img = np.array(img) / 255.0  # Normalize the image

            # Ensure 3 channels (RGB)
            if img.ndim == 2:  # Grayscale image (64, 64)
                img = np.expand_dims(img, axis=-1)  # Shape (64, 64, 1)
                img = np.repeat(img, 3, axis=-1)  # Shape (64, 64, 3)

            # Expand dimensions to add batch axis
            img = np.expand_dims(img, axis=0)  # Shape (1, 64, 64, 3)
            cleaned_images.append(img)

        # Step 5: Perform classification for each resized image
        predictions = []
        for img in cleaned_images:
            try:
                # Predict using TensorFlow model
                print("Making prediction...")
                output = model.predict(img)
                print(f"Prediction result: {output}")
            except Exception as e:
                print(f"Error during prediction: {e}")
                return {"error": str(e)}

            predicted_label_index = np.argmax(output, axis=1)[0]
            predicted_label_name = class_labels.get(predicted_label_index, "Unknown")
            predictions.append(predicted_label_name)

        # Clean up temporary files
        temp_files = [preprocessed_image_path] + segmented_image_paths + resized_image_paths
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"Removed temporary file: {temp_file}")

        return predictions

    except Exception as e:
        return {"error": str(e)}
