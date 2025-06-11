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
