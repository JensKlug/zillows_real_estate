
import os
import glob
import joblib
from colorama import Fore, Style

from zillow.params import MODEL_TARGET, LOCAL_REGISTRY_PATH

def load_model(stage="Production"):
    """
    Return a saved model from the local registry using joblib.
    """

    if MODEL_TARGET == "local":
        print(Fore.YELLOW + f"📁 LOCAL_REGISTRY_PATH is: {LOCAL_REGISTRY_PATH}" + Style.RESET_ALL)

        print(Fore.BLUE + f"\n🔍 Loading latest model from local registry..." + Style.RESET_ALL)

        local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
        local_model_paths = glob.glob(f"{local_model_directory}/*.joblib")

        if not local_model_paths:
            print("❌ No model found in local registry.")
            return None

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

        print(Fore.BLUE + f"📦 Loading model from: {most_recent_model_path_on_disk}" + Style.RESET_ALL)

        model = joblib.load(most_recent_model_path_on_disk)

        print("✅ Model successfully loaded from disk.")
        return model



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
