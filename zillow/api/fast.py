from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from zillow.ml_logic.load_model import load_model
from zillow.ml_logic.data import load_data, create_zip_dict, clean_data, prepare_user_input
from fastapi import HTTPException


# Load cleaned data
data_df = load_data()
cleaned_data_df = clean_data(data_df)

# Create zip_dict at startup
print("Columns in cleaned_data_df:", cleaned_data_df.columns.tolist())
zip_dict = create_zip_dict(cleaned_data_df)
print(f"Loaded {len(zip_dict)} ZIP codes:")
print(list(zip_dict.keys())[:20])

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
    bed: int
    bath: int
    acre_lot: float
    zip_code: str
    house_size: float

@app.post("/predict")
def predict(features: HouseFeatures):
    data = features.model_dump()

    zip_code = data["zip_code"]  # No conversion to int
    if zip_code not in zip_dict:
        return {"error": f"Zip code {zip_code} not found in zip_dict"}

    input_df = prepare_user_input(user_input=data, zip_dict=zip_dict)

    print("Input DataFrame:\n", input_df)  # Debug print

    try:
        prediction = model.predict(input_df)[0]
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing column(s): {e.args[0]}")

    return {"predicted_price": round(float(prediction), 2)}


# Trend Estimate for ZIP_CODE:
class ZIP_CODE(BaseModel):
    time_horizon: int # 3 months, 6 months, 12 months
    zip_code: int

@app.post("/predict_investment")
def predict_investment(features: ZIP_CODE):
    input_df = pd.DataFrame([features.model_dump()])
    prediction = model.predict(input_df)[0]
    return {"predicted_price": round(float(prediction), 2)}
