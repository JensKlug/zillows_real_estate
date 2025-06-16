import traceback
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from zillow.ml_logic.registry import load_model
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
    #raise RuntimeError("❌ Could not load model.")
    print("⚠️ Model not found. API will respond with errors for prediction endpoints.")


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

    zip_code = data["zip_code"]
    if zip_code not in zip_dict:
        raise HTTPException(status_code=400, detail=f"Zip code {zip_code} not found in zip_dict")

    input_df = prepare_user_input(user_input=data, zip_dict=zip_dict)

    print("Input DataFrame (before reordering):\n", input_df)

    try:
        input_df = input_df[model.feature_names_in_]  # Ensure correct column order
        print("Reordered DataFrame:\n", input_df)
        prediction = model.predict(input_df)[0]
        print("Prediction:", prediction)
    except Exception as e:
        import traceback
        traceback.print_exc()  # 👈 shows full stack trace in logs
        raise HTTPException(status_code=500, detail="Internal server error during prediction")

    return {"predicted_price": round(float(prediction), 2)}





# Trend Estimate for ZIP_CODE:




df = pd.read_pickle("/home/jensk/code/JensKlug/zillows_real_estate/raw_data/all_combine.pkl")

# Clean/standardize columns
df.columns = ["zip_code", "result1", "result3", "result6", "result12"]
df["zip_code"] = df["zip_code"].astype(int)

df["zip_code"] = df["zip_code"].astype(int) # Currently the zipcodes are saved as float numbers but integer is better in this case.


class ZIP_CODE(BaseModel):
    time_horizon: int # 1 month, 3 months, 6 months, 12 months
    zip_code: int


@app.post("/predict_investment")
def predict_investment(features: ZIP_CODE):
    zip_code = features.zip_code
    time_horizon = features.time_horizon

    if time_horizon not in [1, 3, 6, 12]:
        raise HTTPException(status_code=400, detail="Only 1, 3, 6, or 12 month horizons are supported")


    row = df[df["zipcode"] == zip_code]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"No data found for ZIP code {zip_code}")


    col_name = f"result{time_horizon}"
    value = int(row.iloc[0][col_name])

    return {
        "zip_code": zip_code,
        "time_horizon_months": time_horizon,
        "is_good_investment": value
    }


# '''
# @app.post("/predict_investment")
# def predict_investment(features: ZIP_CODE):
#     input_df = pd.DataFrame([features.model_dump()])
#     prediction = model.predict(input_df)[0]
#     return {"predicted_price": round(float(prediction), 2)}
# '''
