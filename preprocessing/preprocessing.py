import os
from dotenv import load_dotenv
from utils import process_dataset
from utils import bw_conversion
from utils import detect_edges
from utils import segment_fonts
from utils import resize_with_padding
from utils import analyze_dataset


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(ROOT_DIR)
ENV_PATH = os.path.join(ROOT_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)

RESIZE = os.getenv("RESIZE")
RESIZE = int(RESIZE)
TARGET_SIZE = (RESIZE, RESIZE)

DATA_PATH = os.getenv("DATA_PATH", "").strip()
RAW_DATA_PATH = os.path.join(ROOT_DIR, DATA_PATH, os.getenv("RAW_DATA", "").strip())
GS_DATA_PATH = os.path.join(ROOT_DIR, DATA_PATH, os.getenv("GS_DATA", "").strip())
BW_DATA_PATH = os.path.join(ROOT_DIR, DATA_PATH, os.getenv("BW_DATA", "").strip())
EDGE_DATA_PATH = os.path.join(ROOT_DIR, DATA_PATH, os.getenv("EDGE_DATA", "").strip())
SEGMENTED_FONT_DATA_PATH = os.path.join(ROOT_DIR, DATA_PATH, os.getenv("SEGMENTED_FONT_DATA", "").strip())
RESIZE_DATA_PATH = os.path.join(ROOT_DIR, DATA_PATH, os.getenv("RESIZE_DATA", "").strip())
CLEAN_DATA_PATH = os.path.join(ROOT_DIR, DATA_PATH, os.getenv("CLEAN_DATA", "").strip())
LOG_PATH = os.path.join(ROOT_DIR, os.getenv("LOG_PATH", "").strip())

print(f"Root Directory: {ROOT_DIR}")
print(f"Raw Data Path: {RAW_DATA_PATH}")
print(f"Gray Scale Converted Data Path: {GS_DATA_PATH}")
print(f"Black & White Data Path: {BW_DATA_PATH}")
print(f"Edge Data Path: {EDGE_DATA_PATH}")
print(f"Segmented Data Path: {SEGMENTED_FONT_DATA_PATH}")
print(f"Resized Data Path: {RESIZE_DATA_PATH}")
print(f"Clean Data Path: {CLEAN_DATA_PATH}")
print(f"Log Path: {LOG_PATH}")

if __name__ == "__main__":
    print("========================START========================")
    print(f"Analyze raw dataset: {RAW_DATA_PATH}")
    analyze_dataset(RAW_DATA_PATH, LOG_PATH)
    print("=========================END=========================")

    print("========================START========================")
    print(f"Processing images from: {RAW_DATA_PATH}")
    print(f"Saving processed images to: {GS_DATA_PATH}")
    process_dataset(input_dir=RAW_DATA_PATH, output_dir=GS_DATA_PATH)
    print("Text converted to white complete!")
    print("=========================END=========================")

    print("========================START========================")
    print(f"Processing images from: {GS_DATA_PATH}")
    print(f"Saving processed images to: {BW_DATA_PATH}")
    bw_conversion(input_dir=GS_DATA_PATH, output_dir=BW_DATA_PATH)
    print("Text converted to black complete!")
    print("=========================END=========================")

    print("========================START========================")
    print(f"Processing images from: {BW_DATA_PATH}")
    print(f"Saving processed images to: {EDGE_DATA_PATH}")
    detect_edges(input_dir=BW_DATA_PATH, output_dir=EDGE_DATA_PATH)
    print("Edge detection complete!")
    print("=========================END=========================")

    print("========================START========================")
    print(f"Processing images from: {BW_DATA_PATH}")
    print(f"Saving processed images to: {SEGMENTED_FONT_DATA_PATH}")
    segment_fonts(input_dir=BW_DATA_PATH, output_dir=SEGMENTED_FONT_DATA_PATH)
    print("Font segmentation complete!")
    print("=========================END=========================")

    print("========================START========================")
    print(f"Processing images from: {SEGMENTED_FONT_DATA_PATH}")
    print(f"Saving processed images to: {RESIZE_DATA_PATH}")
    resize_with_padding(input_dir=SEGMENTED_FONT_DATA_PATH, output_dir=RESIZE_DATA_PATH, target_size=TARGET_SIZE)
    print("Resize with padding complete!")
    print("=========================END=========================")

    print("========================START========================")
    print(f"Processing images from: {RESIZE_DATA_PATH}")
    print(f"Saving processed images to: {CLEAN_DATA_PATH}")
    bw_conversion(input_dir=RESIZE_DATA_PATH, output_dir=CLEAN_DATA_PATH)
    print("Data cleaning complete!")
    print("=========================END=========================")

    print("========================START========================")
    print(f"Analyze clean dataset: {CLEAN_DATA_PATH}")
    analyze_dataset(CLEAN_DATA_PATH, LOG_PATH)
    print("=========================END=========================")
