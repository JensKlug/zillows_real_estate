from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from zillow.ml_logic.load_model import load_model


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
    p_c_income: None
    ppsf_zipcode: None

@app.post("/predict")
def predict(features: HouseFeatures):
    input_df = pd.DataFrame([features.model_dump()])
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
