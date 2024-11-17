import os
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from PIL import Image
from tqdm import tqdm


def process_image(image_path):
    """Convert an image to black text on a white background."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image at {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Invert colors
    inverted = cv2.bitwise_not(gray)

    # Apply thresholding for binarization
    _, thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh


def process_dataset(input_dir, output_dir):
    """Process all images in the dataset and save converted images."""
    for root, dirs, files in os.walk(input_dir):
        for file in tqdm(files, desc=f"Processing {root}"):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                # Get input and output file paths
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_dir)
                output_folder = os.path.join(output_dir, relative_path)
                output_path = os.path.join(output_folder, file)

                # Ensure output folder exists
                os.makedirs(output_folder, exist_ok=True)

                try:
                    # Process the image
                    processed_image = process_image(input_path)

                    # Save the processed image
                    cv2.imwrite(output_path, processed_image)
                except Exception as e:
                    print(f"Error processing {input_path}: {e}")


def segment_fonts(input_dir, output_dir):
    """Segment words from images and save them in their labeled directories."""
    for root, dirs, files in os.walk(input_dir):
        for file in tqdm(files, desc=f"Processing {root}"):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(root, file)

                relative_path = os.path.relpath(root, input_dir)
                output_folder = os.path.join(output_dir, relative_path)
                os.makedirs(output_folder, exist_ok=True)

                try:
                    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
                    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    for i, contour in enumerate(contours):
                        x, y, w, h = cv2.boundingRect(contour)
                        segmented_word = binary_image[y:y + h, x:x + w]

                        word_file_name = f"{os.path.splitext(file)[0]}_font_{i}.png"
                        word_output_path = os.path.join(output_folder, word_file_name)
                        cv2.imwrite(word_output_path, segmented_word)
                except Exception as e:
                    print(f"Error processing {input_path}: {e}")


def resize_with_padding(input_dir, output_dir, target_size=(128, 128)):
    os.makedirs(output_dir, exist_ok=True)
    for root, dirs, files in os.walk(input_dir):
        for file in tqdm(files, desc=f"Processing {root}"):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_dir)
                output_folder = os.path.join(output_dir, relative_path)
                os.makedirs(output_folder, exist_ok=True)

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
                    print(f"Skipping image {file} with invalid dimensions: {img.size}")
                    continue

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
                    print(f"Skipping image {file} with invalid resized dimensions: ({new_width}, {new_height})")
                    continue

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
                padded_img.save(os.path.join(output_folder, file))


def log_sample_distribution(dataset_path, log_file):
    """
    Log the number of samples for each label in the dataset.
    """
    label_counts = defaultdict(int)
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                label = os.path.basename(root)  # Use folder name as label
                label_counts[label] += 1

    log_file.write("\nDataset Summary:\n")
    log_file.write(f"Total Labels: {len(label_counts)}\n")
    for label, count in label_counts.items():
        log_file.write(f"Label: {label}, Samples: {count}\n")

    return label_counts


def log_image_dimensions(dataset_path, log_file):
    """
    Log image dimensions from the dataset.
    """
    dimensions = []
    for root, dirs, files in os.walk(dataset_path):
        for file in tqdm(files, desc=f"Processing {root}"):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                image = cv2.imread(img_path)
                if image is not None:
                    h, w = image.shape[:2]
                    dimensions.append((w, h))
                else:
                    log_file.write(f"Warning: Unable to read image at {img_path}\n")

    log_file.write("\nImage Dimensions:\n")
    for w, h in dimensions:
        log_file.write(f"Width: {w}, Height: {h}\n")

    return dimensions


def plot_sample_distribution(label_counts, output_dir, dataset_path):
    """
    Plot and save the sample distribution for each label.
    """
    file_name = f"{output_dir}/{dataset_path}_sample_dis.png"
    os.makedirs(output_dir, exist_ok=True)
    labels = list(label_counts.keys())
    counts = list(label_counts.values())

    plt.figure(figsize=(12, 6))
    plt.bar(labels, counts, color='skyblue')
    plt.xlabel('Labels')
    plt.ylabel('Number of Samples')
    plt.title('Number of Samples Per Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(file_name)
    plt.show()


def plot_image_dimension_statistics(dimensions, output_dir, dataset_path):
    """
    Plot and save the statistics of image dimensions.
    """
    file_name = f"{output_dir}/{dataset_path}_image_dim.png"
    os.makedirs(output_dir, exist_ok=True)
    heights = [h for _, h in dimensions]
    widths = [w for w, _ in dimensions]

    max_height = max(heights)
    min_height = min(heights)
    avg_height = sum(heights) / len(heights)

    max_width = max(widths)
    min_width = min(widths)
    avg_width = sum(widths) / len(widths)

    stats = ['Max', 'Min', 'Avg']
    height_values = [max_height, min_height, avg_height]
    width_values = [max_width, min_width, avg_width]

    plt.figure(figsize=(8, 6))
    x = range(len(stats))
    plt.bar(x, height_values, width=0.4, label='Height', color='skyblue', align='center')
    plt.bar([i + 0.4 for i in x], width_values, width=0.4, label='Width', color='salmon', align='center')
    plt.xticks([i + 0.2 for i in x], stats)
    plt.xlabel('Statistics')
    plt.ylabel('Pixels')
    plt.title('Image Dimension Statistics')
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_name)
    plt.show()


def analyze_dataset(dataset_path, log_dir_path):
    """
    Main function to analyze the dataset by combining logging and plotting.
    """
    dataset_name = os.path.basename(dataset_path)
    os.makedirs(log_dir_path, exist_ok=True)

    sample_distribution_log_file_path = f"{log_dir_path}/{dataset_name}_sample_dis_analysis.log"
    with open(sample_distribution_log_file_path, "w") as log_file:
        log_file.write("Sample Distribution Analysis Log\n")
        log_file.write("=" * 50 + "\n")

        print("Logging sample distribution...")
        label_counts = log_sample_distribution(dataset_path, log_file)

    image_dimension_log_file_path = f"{log_dir_path}/{dataset_name}_image_dim_analysis.log"
    with open(image_dimension_log_file_path, "w") as log_file:
        log_file.write("Image Dimension Analysis Log\n")
        log_file.write("=" * 50 + "\n")
        print("Logging image dimensions...")
        dimensions = log_image_dimensions(dataset_path, log_file)

    print("Plotting sample distribution...")
    plot_sample_distribution(label_counts, log_dir_path, dataset_name)

    print("Plotting image dimension statistics...")
    plot_image_dimension_statistics(dimensions, log_dir_path, dataset_name)

    print(f"Analysis log saved to: {log_dir_path}")
    print(f"Plots saved to: {log_dir_path}")


def remove_black_images_in_dir(input_dir, black_threshold=0.6):
    """
    Remove images in the given directory (and subdirectories) if more than
    `black_threshold` (default 80%) of the pixels are black.

    Args:
    - input_dir (str): Path to the directory containing label subdirectories.
    - black_threshold (float): The threshold for black pixel percentage (default 0.8 for 80%).
    """
    # Walk through the directory
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_path = os.path.join(root, file)

                # Read the image in grayscale
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                # Check if the image is loaded correctly
                if image is None:
                    continue

                # Calculate the total number of pixels in the image
                total_pixels = image.size

                # Count the number of black pixels (pixels with value 0)
                black_pixels = np.sum(image == 0)

                # Calculate the percentage of black pixels
                black_percentage = black_pixels / total_pixels

                # If black pixels exceed the threshold, remove the image
                if black_percentage > black_threshold:
                    print(f"Removing {image_path} due to {black_percentage * 100:.2f}% black pixels.")
                    os.remove(image_path)
