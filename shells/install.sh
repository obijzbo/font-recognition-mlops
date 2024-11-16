ENV_FILE=".env"

if [ -f $ENV_FILE ]; then
  source $ENV_FILE
  echo "File Loaded: $ENV_FILE"
else
  echo "$ENV_FILE File Not Found"
  echo "Make sure $ENV_FILE exist and try again!!!"
  exit 1
fi

echo "python version: $PYTHON_VERSION"
echo "environment name: $ENV_NAME"

check_python_version() {
    if command -v python${PYTHON_VERSION} >/dev/null 2>&1; then
        echo "python${PYTHON_VERSION} is already installed."
        sudo apt install -y python${PYTHON_VERSION}-venv
        return 0
    else
        echo "python${PYTHON_VERSION} is not installed."
        return 1
    fi
}

install_python_version() {
    echo "Installing Python ${PYTHON_VERSION}..."
    sudo apt update
    sudo apt install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt update
    sudo apt install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-venv
}

#install_tesseract_ocr() {
#  echo "Tesseract Installation ..."
#  sudo apt update
#  sudo apt install -y tesseract-ocr
#}

create_virtual_env() {
    echo "Creating a virtual environment with python${PYTHON_VERSION}"
    python${PYTHON_VERSION} -m venv ${ENV_NAME}
    source ${ENV_NAME}/bin/activate

    # Install packages from requirements.txt
    if [ -f "preprocessing_req.txt" ]; then
        echo "Installing packages from requirements.txt..."
        pip3 install --upgrade pip
        pip3 install -r preprocessing_req.txt
    else
        echo "requirements.txt not found. Skipping package installation."
    fi

    deactivate
}


echo "Checking for Python ${PYTHON_VERSION}..."
if ! check_python_version; then
    install_python_version
fi

create_virtual_env

echo "Setup complete!!!"