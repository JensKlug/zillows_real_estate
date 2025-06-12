from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from zillow.ml_logic.load_model import load_model
from zillow.ml_logic.data import load_data, create_zip_dict, clean_data, prepare_user_input

# Load cleaned data
data_df = load_data()
cleaned_data_df = clean_data(data_df)

# Create zip_dict at startup
zip_dict = create_zip_dict(cleaned_data_df)

app = FastAPI()

model = load_model()
if model is None:
    raise RuntimeError("❌ Could not load model.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Welcome to the House Price Predictor!"}


class HouseFeatures(BaseModel):
    house_size: float
    bed: float
    bath: float
    acre_lot: float
    zip_code: int

@app.post("/predict")
def predict(features: HouseFeatures):
    # Convert input to dict
    data = features.model_dump()

    # Look up zip_code in dictionary
    zip_code = int(data["zip_code"])
    if zip_code not in zip_dict:
        return {"error": f"Zip code {zip_code} not found in zip_dict"}

    # Create DataFrame with all required model features
    input_df = prepare_user_input(user_input=data, zip_dict=zip_dict)

    # # Predict
    # prediction = model.predict(input_df)[0]
    print("Input DataFrame:\n", input_df)  # <-- Debug print

    try:
        prediction = model.predict(input_df)[0]
    except KeyError as e:
        missing_cols = e.args[0]
        raise HTTPException(status_code=400, detail=f"Missing column(s): {missing_cols}")

    return {"predicted_price": round(float(prediction), 2)}
