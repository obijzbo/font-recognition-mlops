import os
from dotenv import load_dotenv

from utils import create_directory_if_not_exists
from utils import track_version
from utils import remove_directory_if_exists
from utils import split_dataset
from utils import model_performance_track

from imageprocessing import process_dataset
from imageprocessing import segment_fonts
from imageprocessing import resize_with_padding
from imageprocessing import analyze_dataset

from model_training import train_model_lenet5

from evaluation import evaluate_model


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(ROOT_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)

RESIZE = os.getenv("RESIZE")
RESIZE = int(RESIZE)
TARGET_SIZE = (RESIZE, RESIZE)

DATA_PATH = os.getenv("DATA_PATH", "").strip()
VERSION_DIR = os.path.join(ROOT_DIR, "versions")

MODEL_PERFORMANCE_LOG = os.path.join(ROOT_DIR, os.getenv("MODEL_LOG_FILE", "").strip())

if __name__ == "__main__":
    print("=====================Versioning=====================")
    version_num = track_version(VERSION_DIR)
    print(version_num)
    current_version_dir = f"version_{version_num}"
    CURRENT_VERSION_PATH = os.path.join(VERSION_DIR, current_version_dir)
    create_directory_if_not_exists(CURRENT_VERSION_PATH)
    print(f"=====================Version {version_num}=====================")

    # CURRENT_VERSION_PATH = '/home/mtlls1/Projects/font-recognition-mlops/versions/version_3'

    RAW_DATA_PATH = os.getenv("RAW_DATA_PATH")
    GS_DATA_PATH = os.path.join(CURRENT_VERSION_PATH, os.getenv("GS_DATA", "").strip())
    SEGMENTED_FONT_DATA_PATH = os.path.join(CURRENT_VERSION_PATH, os.getenv("SEGMENTED_FONT_DATA", "").strip())
    RESIZE_DATA_PATH = os.path.join(CURRENT_VERSION_PATH, os.getenv("RESIZE_DATA", "").strip())
    CLEAN_DATA_PATH = os.path.join(CURRENT_VERSION_PATH, os.getenv("CLEAN_DATA", "").strip())
    TRAIN_DATA_PATH = os.path.join(CURRENT_VERSION_PATH, os.getenv("TRAIN_PATH", "").strip())
    TEST_DATA_PATH = os.path.join(CURRENT_VERSION_PATH, os.getenv("TEST_PATH", "").strip())
    VAL_DATA_PATH = os.path.join(CURRENT_VERSION_PATH, os.getenv("VAL_PATH", "").strip())

    print("=====================Pipeline Execution START=====================")

    print("==================================================================")
    print("==================================================================")

    print("=====================STEP 1: DATA Preprocessing START=====================")

    print("========================START========================")
    print(f"Analyze raw dataset: {RAW_DATA_PATH}")
    analyze_dataset(RAW_DATA_PATH, CURRENT_VERSION_PATH)
    print("=========================END=========================")

    print("========================START========================")
    print(f"Processing images from: {RAW_DATA_PATH}")
    print(f"Saving processed images to: {GS_DATA_PATH}")
    process_dataset(input_dir=RAW_DATA_PATH, output_dir=GS_DATA_PATH)
    print("Text converted to white complete!")
    print("=========================END=========================")

    print("========================START========================")
    print(f"Processing images from: {GS_DATA_PATH}")
    print(f"Saving processed images to: {SEGMENTED_FONT_DATA_PATH}")
    segment_fonts(input_dir=GS_DATA_PATH, output_dir=SEGMENTED_FONT_DATA_PATH)
    remove_directory_if_exists(GS_DATA_PATH)
    print("Font segmentation complete!")
    print("=========================END=========================")

    print("========================START========================")
    print(f"Processing images from: {SEGMENTED_FONT_DATA_PATH}")
    print(f"Saving processed images to: {RESIZE_DATA_PATH}")
    resize_with_padding(SEGMENTED_FONT_DATA_PATH, RESIZE_DATA_PATH, target_size=TARGET_SIZE)
    remove_directory_if_exists(SEGMENTED_FONT_DATA_PATH)
    print("Resize with padding complete!")
    print("=========================END=========================")

    print("========================START========================")
    print(f"Processing images from: {RESIZE_DATA_PATH}")
    print(f"Saving processed images to: {CLEAN_DATA_PATH}")
    process_dataset(input_dir=RESIZE_DATA_PATH, output_dir=CLEAN_DATA_PATH)
    remove_directory_if_exists(RESIZE_DATA_PATH)
    print("Data cleaning complete!")
    print("=========================END=========================")

    print("========================START========================")
    print(f"Analyze clean dataset: {CLEAN_DATA_PATH}")
    analyze_dataset(CLEAN_DATA_PATH, CURRENT_VERSION_PATH)
    print("=========================END=========================")

    print("======================STEP 1: DATA Preprocessing END======================")

    print("==================================================================")
    print("==================================================================")

    print("=====================STEP 2: DATA SPLIT START=====================")

    print("========================START========================")
    split_dataset(CLEAN_DATA_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH, VAL_DATA_PATH, train_ratio=0.7, val_ratio=0.15,
                  test_ratio=0.15)
    remove_directory_if_exists(CLEAN_DATA_PATH)
    print("=========================END=========================")

    print("======================STEP 2: DATA SPLIT END======================")

    print("==================================================================")
    print("==================================================================")

    print("=====================STEP 3: MODEL TRAINING START=====================")

    print("========================START========================")
    train_model_lenet5(TRAIN_DATA_PATH, TEST_DATA_PATH, CURRENT_VERSION_PATH)
    print("=========================END=========================")

    print("======================STEP 3: MODEL TRAINING END======================")

    print("==================================================================")
    print("==================================================================")

    print("=====================STEP 4: MODEL EVALUATION START=====================")

    print("========================START========================")
    model_accuracy = evaluate_model("lenet5_model.keras", "class_labels.json", TEST_DATA_PATH, CURRENT_VERSION_PATH)
    model_performance_track(MODEL_PERFORMANCE_LOG, f"{CURRENT_VERSION_PATH}/lenet5_model.keras", model_accuracy, f"version_{version_num}")
    print("=========================END=========================")

    print("======================STEP 4: MODEL EVALUATION END======================")

    print("==================================================================")
    print("==================================================================")

    print("======================Pipeline Execution END======================")
