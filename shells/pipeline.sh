#!/bin/bash

ENV_FILE=".env"

if [ -f $ENV_FILE ]; then
  source $ENV_FILE
  echo "File Loaded: $ENV_FILE"
else
  echo "$ENV_FILE File Not Found"
  echo "Make sure $ENV_FILE exist and try again!!!"
  exit 1
fi

# Get the absolute path of the current script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

"$SCRIPT_DIR/install.sh"

source venv/bin/activate

 Check if the virtual environment was activated successfully
if [ $? -ne 0 ]; then
    echo "Failed to activate the virtual environment."
    exit 1
fi

cd "$(dirname "$0")/../pipeline"

python3 main.py

echo "File Name: $MODEL_LOG_FILE"

model_file_path=$(jq '.model_file_path' "$MODEL_LOG_FILE")
version=$(jq -r '.version' "$MODEL_LOG_FILE")

echo "Model File Path: $model_file_path"
echo "Version: $version"

# Trim leading/trailing whitespace or quotes
model_file_path=$(echo "$model_file_path" | sed 's/^"//;s/"$//')

cp $model_file_path app/
cp versions/$version/class_labels.json app/

echo "Pipeline Execution Complete"
echo "Deploy the project locally from app/ directory"
echo "Run the docker compose with --build flag"