
import os
import glob
import joblib
from colorama import Fore, Style

from zillow.params import MODEL_TARGET, LOCAL_REGISTRY_PATH

import os
import joblib
from pathlib import Path

def load_model(model_path=None):
    """
    Load the trained XGBoost model from a specified path or the default registry location.

    Args:
        model_path (str, optional): Path to the model file. If None, uses the default local registry path.

    Returns:
        model: Loaded XGBoost model, or None if not found.
    """
    # Default registry path if not specified
    if model_path is None:
        local_registry = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model')
        model_path = os.path.join(local_registry, 'xgboost_best_model.pkl')

    # Check if the path exists
    if not os.path.exists(model_path):
        print(f"❌ No model found at {model_path}")
        return None

    try:
        model = joblib.load(model_path)
        print(f"✅ Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"❌ Error loading model from {model_path}: {str(e)}")
        return None


def save_model(model, path="model/xgboost_best_model.pkl"):
    """
    Save the trained model to a specified path and log the action.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(Fore.GREEN + f"✅ Model saved to {path}" + Style.RESET_ALL)
    return path
# def load_model(path="model/xgboost_best_model.pkl"):
#     return joblib.load(path)
print("test")
