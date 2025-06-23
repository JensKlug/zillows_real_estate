
import os
import glob
import time
import joblib
from colorama import Fore, Style
from zillow.params import MODEL_TARGET, LOCAL_REGISTRY_PATH
from pathlib import Path
from datetime import datetime

def load_model():
    """
    Load the most recently saved XGBoost model from the 'models' directory.

    The function performs the following steps:
        - Determines the path to the project-level 'models' directory relative to the current script.
        - Searches for all model files matching the pattern 'xgboost*.pkl'.
        - Loads the most recently created model file using `joblib`.
        - Prints informative status messages during the process.

    Returns:
        XGBRegressor or None: The loaded XGBoost model object if successful;
        otherwise, returns None if no model is found or if loading fails.
    """

    # Always get the directory of the *current script file*
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Go up two levels to reach the root project directory
    project_root = os.path.abspath(os.path.join(base_dir, '..', '..'))
    models_dir = os.path.join(project_root, 'models')

    print(f"Looking for models in: {models_dir}")
    model_files = glob.glob(os.path.join(models_dir, 'xgboost*.pkl'))
    print(f"Found model files: {model_files}")
    if not model_files:
        print("❌ No model files found")
        return None
    latest_model = max(model_files, key=os.path.getctime)
    try:
        model = joblib.load(latest_model)
        print(f"✅ Model loaded from {latest_model}")
        return model
    except Exception as e:
        print(f"❌ Error loading model from {latest_model}: {str(e)}")
        return None

def save_model(model) -> str:
    """
    Save a trained XGBoost model to the 'models' directory with a timestamped filename.

    The function performs the following steps:
        - Generates a timestamped filename using the current date and time.
        - Saves the model to the 'models' directory using `joblib.dump()`.
        - Prints the path where the model was saved.

    Returns:
        str: The file path to the saved model (e.g., 'models/xgboost_model_YYYYMMDD_HHMMSS.pkl').

    Args:
        model (XGBRegressor): Trained XGBoost model to be saved.
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join("models", f"xgboost_model_{timestamp}.pkl")
    joblib.dump(model, model_path)  # ✅ scikit-learn compatible saving
    print(f"✅ Model saved to {model_path}")
    return model_path
