from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset path
DATA_PATH = PROJECT_ROOT / "data" / "heart_disease_processed.csv"

# Directory for artifacts
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

# MLflow experiment name
EXPERIMENT_NAME = "Heart Disease Classification"

# MLflow tracking URI
TRACKING_URI = "http://127.0.0.1:5000"

# Target column in the dataset
TARGET_COLUMN = "target"

# Random state for reproducibility
RANDOM_STATE = 42