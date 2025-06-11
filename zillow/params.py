import os

# Environment variables or defaults
MODEL_TARGET = os.getenv("MODEL_TARGET", "local")
LOCAL_REGISTRY_PATH = os.path.join("zillow", "model")
