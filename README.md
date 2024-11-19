# font-recognition-mlops
End to end ML pipeline to train and deploy font recognition model locally

## Problem statement
Train a model that can classify fonts.

## Intuition
The dataset contains around 50 classes, each with around 200 samples. Each samples are an image with a single line. Each line consist of a single word, or multiple words. But the words or the lines are made up with arbitary letter, numeric and alphabetical. 

As the words or lines have any contextual value, we can choose to focus only on the letters to extract features of a specific font. 

Also, in the dataset there is a class which contains blank white images.

Getting back to the point of focusing on letters to classify font, it aligns with the MNIST hand written data recognition problem. 

And as we are only focusing on the fonts, we extracted each font from the dataset, binarized the image, and then train for classification. 

Reason to do so is to reduce complexity. 

And as the data were already augmented, and converted data are binarized, during traing data were not heavily augmented.

And as we were addressing it as the MNIST hand written data recognition problem, we decided to go with simple CNN. As implemented ResNet, VGG would be an overkill for binarized image classification. 

## How to run this project?

### Environment
OS - Ubuntu 22.04 and above

### Project Structure
The project has multiple python packages.
- app - This portion contains the fastapi code with Dockerfile and docker compose to serve the model
- pipeline - This portion contains the main pipeline that will preprocess the dataset for model training, train the model, evaluate it, log performance, and track best performing model
- preprocessing - experimental section to try and test data preprocessing steps to decide which one works better
- training - experimental section to try and test various models to decide which one works better
- versions - This directory contains artifact of each pipeline iteration
- model_performance.json - This file tracks the best performing version number and model path
- shells - Contains the shell script to setup the project and execute the pipeline

### Project Setup and Execution
The project have two environment file. One should be placed on the project root, another inside the app directory. 

To execute the pipeline, ensure .env is in place. Then execute the pipeline.sh from the project root

`./shells/pipeline.sh`

If you want to trun of the project setup step, comment line 17 in pipeline.sh

Once the pipeline execution is complete, go to app directory

`cd app`

Then execute the API with

`docker compose down && docker compose up --build`

As the project will be run and deployed locally, inside docker compose, docker will build from the file context.

API deployment has not been attached with the pipeline to ensure human confirmation on which model should be deployed.

Inside the dockerfile, tensorflow is used as base image to reduce docker build time.


## Contact
If you have any questions regarding this project, feel free to reach out to me.

Email - nahiyanmubashshir@gmail.com
