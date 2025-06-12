from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from zillow.ml_logic.load_model import load_model
from zillow.ml_logic.data import load_data, create_zip_dict

# Load cleaned data
cleaned_df = load_data()

# Create zip_dict at startup
zip_dict = create_zip_dict(cleaned_df)

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

# Price Estimate according to Real Estate Basic data.
class HouseFeatures(BaseModel):
    house_size: float
    bed: float
    bath: float
    acre_lot: float
    zip_code: float

@app.post("/predict")
def predict(features: HouseFeatures):
    # Convert input to dict
    data = features.model_dump()

    # Look up zip_code in dictionary
    zip_code = int(data["zip_code"])
    if zip_code not in zip_dict:
        return {"error": f"Zip code {zip_code} not found in zip_dict"}

    # Add looked-up values to input data
    data["p_c_income"] = zip_dict[zip_code][0]
    data["ppsf_zipcode"] = zip_dict[zip_code][1]

    # Create DataFrame with all required model features
    input_df = pd.DataFrame([data])

    # Predict
    prediction = model.predict(input_df)[0]

    return {"predicted_price": round(float(prediction), 2)}


# Trend Estimate for ZIP_CODE:
class ZIP_CODE(BaseModel):
    time_horizon: int # 3 months, 6 months, 12 months
    zip_code: int

@app.post("/predict")
def predict_investment(features: ZIP_CODE):
    input_df = pd.DataFrame([features.model_dump()])
    prediction = model.predict(input_df)[0]
    return {"predicted_price": round(float(prediction), 2)}
