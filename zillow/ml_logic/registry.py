
import os
import glob
import time
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



# def save_model(model):
#     """
#     Save the trained model to a specified path and log the action.
#     """
#     timestamp = time.strftime("%Y%m%d-%H%M%S")

#     # Save model locally
#     model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{timestamp}.h5")
#     model.save(model_path)

#     os.makedirs(os.path.dirname(model_path), exist_ok=True)
#     joblib.dump(model, path)
#     print(Fore.GREEN + f"✅ Model saved to {model_path}" + Style.RESET_ALL)
#     return model_path

def save_model(model) -> str:
    from datetime import datetime
    import os
    import joblib

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join("models", f"xgboost_model_{timestamp}.pkl")
    joblib.dump(model, model_path)  # ✅ scikit-learn compatible saving
    print(f"✅ Model saved to {model_path}")
    return model_path
