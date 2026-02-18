# sharp_project
A ML-OPs project. SHARP (Smart Hand Automated Recognition Project) recognises fingers combinations directly on the camera

# Installation and configuration
You can install this project by cloning it or dowloading it.

## Installing requirements
Install using pip the requirements (directly or in a virtual environnement) using the following while being in the project's root :
```bash
pip install -r requirements.txt
```

## Setting up the .env file
in the config/ directory, create a file named '.env' containing the picsellia api token (PICSELLIA_TOKEN) and the path to store the mlflow data (MLFLOW_TRACKING_URI). By default, this path may be ./runs/mlflow.

# Meta data store, Model registry
All is stored locally, and can be viewed with MLFlow using the command (while at the project's root):
```bash
mlflow ui --backend-store-uri file:./mlruns
```
Then, it can be accessed with http://127.0.0.1:5000/

# Data pipeline, Model pipeline
Configuration of all 3 types of training is located at training/src/train_config.py. Each parameter can be changed to your linking. Executing the main.py script at the project's root give you 4 options :

## 1. Download dataset
Download the dataset from picsellia. The configuration for picsellia is in config.py, you must change it to your own dataset. This will download the dataset locally. May take some time.

## 2. Validate Dataset
Run various validity checks on the dataset. Use this to confirm the first step was successfull.

## 3. Prepare Dataset
Prepare the dataset for training, splitting it and treating the data

## 4. Run Training
Run the chosed config. You can provide options to override base config options. You can then choose between the three presets provided. The training may of course take some time, depending on the configuration and chosed model.

# Serving
You can change the model used by the final application by replacing the 'best.pt' file located in the serving/ directory with the one of your choosing (these files are generated during training and stored in the mlruns/ directory).

To run the application, you must be located inside the serving/ directory. Then, use the following command :
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
To run the app. It will be available in a navigator using the url http://localhost:8000