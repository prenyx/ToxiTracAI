import os

# Define the root of the project directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Define subdirectories
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")

# Define specific model paths
KERAS_MODEL_PATH = os.path.join(MODELS_DIR, "heart_nn_normalrate_model2.keras")
SCALER_MODEL_PATH = os.path.join(MODELS_DIR, "scaler_nn_model2.joblib")
