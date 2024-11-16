import os
from dotenv import load_dotenv
from utils import split_dataset
from utils import augment_training_data
from utils import train_font_classification_model
from utils import evaluate_model_on_test_data


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(ROOT_DIR)
ENV_PATH = os.path.join(ROOT_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)

SPLIT_RATIO = os.getenv("SPLIT_RATIO")
SPLIT_RATIO = float(SPLIT_RATIO)

DATA_PATH = os.getenv("DATA_PATH", "").strip()
CLEAN_DATA_PATH = os.path.join(ROOT_DIR, DATA_PATH, os.getenv("CLEAN_DATA", "").strip())
TRAIN_DATA_PATH = os.path.join(ROOT_DIR, DATA_PATH, os.getenv("TRAIN_PATH", "").strip())
TEST_DATA_PATH = os.path.join(ROOT_DIR, DATA_PATH, os.getenv("TEST_PATH", "").strip())
LOG_PATH = os.path.join(ROOT_DIR, os.getenv("LOG_PATH", "").strip())
MODEL_PATH = os.path.join(ROOT_DIR, os.getenv("MODEL_PATH", "").strip())

if __name__ == "__main__":
    # print("========================START========================")
    # print(f"Prepare training and testing dataset ...")
    # split_dataset(CLEAN_DATA_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH, SPLIT_RATIO)
    # print("=========================END=========================")

    # print("========================START========================")
    # print(f"Augment training data ...")
    # augment_training_data(TRAIN_DATA_PATH)
    # print("=========================END=========================")

    # print("========================START========================")
    # print(f"Train Classification Model ...")
    # train_font_classification_model(TRAIN_DATA_PATH, MODEL_PATH, LOG_PATH)
    # print("=========================END=========================")

    print("========================START========================")
    print(f"Evaluate on test data ...")
    evaluate_model_on_test_data("/home/ubuntu/Projects/font-recognition-mlops/MODEL/font_classification_model_20241116-213513.keras", TEST_DATA_PATH, LOG_PATH)
    print("=========================END=========================")
