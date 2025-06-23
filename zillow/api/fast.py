import os
import traceback
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from zillow.ml_logic.registry import load_model
from zillow.ml_logic.data import get_df_yearly_data, load_data, create_zip_dict, clean_data, prepare_user_input, get_df_one_city, get_df_all_cities
from fastapi import HTTPException
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import pickle
import joblib
from fastapi import Body

from zillow.ml_logic.data import df_for_zipcode_graph

#evrard
from pydantic import BaseModel

# Set base directory and project root
base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(base_dir, '..', '..'))


# Get the zipcode directory
zip_dir = os.path.join(project_root, 'raw_data',"zip_dict.pkl")
with open(zip_dir, "rb") as file:
    zip_dict = pickle.load(file)
print(f"Loaded {len(zip_dict)} ZIP codes:")
print(list(zip_dict.keys())[:20])

# Load CSV before app starts
csv_path = os.path.join(project_root, 'raw_data', 'HouseTS.csv')
try:
    house_TS_df = pd.read_csv(csv_path)
    print(f"✅ Loaded median_prices.csv with shape: {house_TS_df.shape}")
except Exception as e:
    print(f"❌ Failed to load median_prices.csv: {e}")


# Start api
app = FastAPI()

#Load model from google cloud console
model = load_model()
preprocessor = joblib.load("models/preprocessor.pkl")
print("✅ Preprocessor loaded.")
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

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    try:
        body = await request.json()
    except:
        body = "Could not read body"
    tb = traceback.format_exc()

    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "body": body,
            "traceback": tb,
            "message": "Validation failed. Check the request structure and field names."
        },
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
    """
    Endpoint to predict house price based on user-provided features.

    This function performs the following:
        - Parses and validates input features using the HouseFeatures model.
        - Checks if the provided zip code exists in the `zip_dict`.
        - Prepares input data into a model-ready DataFrame using `prepare_user_input`.
        - Transforms the data using the pre-fitted preprocessor.
        - Runs the model to generate a price prediction.
        - Returns the predicted price as a JSON response.

    Args:
        features (HouseFeatures): Input data model with attributes like
            'bed', 'bath', 'acre_lot', 'zip_code', 'house_size'.

    Returns:
        dict: JSON response containing the predicted house price (rounded to two decimals).

    Raises:
        HTTPException:
            - 400 if the zip code is not in `zip_dict`.
            - 500 if an internal error occurs during prediction.
    """

    data = features.model_dump()

    zip_code = data["zip_code"]
    if zip_code not in zip_dict:
        raise HTTPException(status_code=400, detail=f"Zip code {zip_code} not found in zip_dict")

    input_df = prepare_user_input(user_input=data, zip_dict=zip_dict)

    try:
        input_scaled = preprocessor.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        print("Prediction:", prediction)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error during prediction")

    return {"predicted_price": round(float(prediction), 2)}


# Trend Estimate for ZIP_CODE:
forecast = os.path.join(project_root, 'raw_data',"all_combine.pkl")
df = pd.read_pickle(forecast)

# Clean/standardize columns
df.columns = ["zip_code", "result1", "result3", "result6", "result12"]
df["zip_code"] = df["zip_code"].astype(int)


class ZIP_CODE(BaseModel):
    time_horizon: int # 1 month, 3 months, 6 months, 12 months
    zip_code: int


@app.post("/predict_investment")
def predict_investment(features: ZIP_CODE):
    """
    Predict whether a given ZIP code is a good real estate investment over a specified time horizon.

    The endpoint performs the following steps:
        - Validates that the requested time horizon is one of the supported options: 1, 3, 6, or 12 months.
        - Checks if data exists for the given ZIP code in the dataset.
        - Retrieves the investment prediction result corresponding to the specified time horizon.
        - Returns the ZIP code, time horizon, and a binary indicator whether it is a good investment.

    Args:
        features (ZIP_CODE): Input model containing:
            - zip_code (str): Five-digit ZIP code to query.
            - time_horizon (int): Investment horizon in months (1, 3, 6, or 12).

    Raises:
        HTTPException 400: If the time horizon is not one of the supported values.
        HTTPException 404: If no data exists for the given ZIP code.

    Returns:
        dict: JSON response containing:
            - zip_code (str): Queried ZIP code.
            - time_horizon_months (int): Investment horizon in months.
            - is_good_investment (int): Binary indicator (e.g., 1 or 0) denoting investment quality.
    """

    zip_code = features.zip_code
    time_horizon = features.time_horizon

    if time_horizon not in [1, 3, 6, 12]:
        raise HTTPException(status_code=400, detail="Only 1, 3, 6, or 12 month horizons are supported")


    row = df[df["zip_code"] == zip_code]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"No data found for ZIP code {zip_code}")


    col_name = f"result{time_horizon}"
    value = int(row.iloc[0][col_name])

    return {
        "zip_code": zip_code,
        "time_horizon_months": time_horizon,
        "is_good_investment": value
    }

class ZipRequest(BaseModel):
    zip_code: str

class ZipRequest(BaseModel):
    zip_code: str

@app.post("/zipcode_trend")
def zipcode_trend(payload: ZipRequest):
    """
    Retrieve the historical price trend for a specified ZIP code.

    The endpoint performs the following steps:
        - Validates that the ZIP code is provided and not empty.
        - Cleans the full housing time series DataFrame using `df_for_zipcode_graph`.
        - Filters the cleaned DataFrame for the requested ZIP code.
        - Returns a 404 response if no data is found for the ZIP code.
        - Sorts the data by date and formats the dates as 'YYYY-MM-DD'.
        - Returns the ZIP code and a list of date-price pairs representing the price trend.

    Args:
        payload (ZipRequest): Input model containing:
            - zip_code (str): Five-digit ZIP code string for which to fetch the trend.

    Raises:
        HTTPException 400: If the ZIP code is missing or empty.
        JSONResponse 404: If no data is found for the given ZIP code.

    Returns:
        dict: JSON response with:
            - zip_code (str): The queried ZIP code.
            - trend (list of dict): List of records with 'date' (str) and 'price' (float) keys representing the price trend.
    """

    zip_code = payload.zip_code.strip()

    if not zip_code:
        raise HTTPException(status_code=400, detail="Missing ZIP code")

    # Clean the full DataFrame using the central function
    cleaned_df = df_for_zipcode_graph(house_TS_df)

    # Filter by ZIP code
    filtered = cleaned_df[cleaned_df["zipcode"] == zip_code]

    if filtered.empty:
        return JSONResponse(content={"message": f"No data found for ZIP {zip_code}"}, status_code=404)

    # Sort and format dates
    filtered = filtered.sort_values("date").copy()
    filtered["date"] = filtered["date"].dt.strftime("%Y-%m-%d")

    return {
        "zip_code": zip_code,
        "trend": filtered[["date", "price"]].to_dict(orient="records")
    }

@app.get('/yearly_price_evolution')
def yearly_price_evolution(zip_code: str):
    """
    Retrieve the yearly average sale price evolution for a given ZIP code.

    The endpoint performs the following steps:
        - Converts date columns to datetime and extracts the year.
        - Groups data by city, ZIP code, and year to compute mean 'median_sale_price' and 'Median Home Value'.
        - Filters the grouped data to return only the records for the specified ZIP code.

    Args:
        zip_code (str): Five-digit ZIP code string for which to fetch yearly price evolution.

    Returns:
        dict: JSON response containing:
            - data (list of dict): List of yearly price records with keys
            ['city', 'zipcode', 'year', 'median_sale_price', 'Median Home Value'].
    """

    df_yearly = get_df_yearly_data(house_TS_df, zip_code)
    return {'data': df_yearly.to_dict('records')}


@app.get('/filter_city')
def filter_city(zip_code: str):
    """
    Retrieve daily average price data for the city corresponding to a given ZIP code.

    The endpoint performs the following steps:
        - Converts ZIP codes to standardized 5-digit strings.
        - Finds the city associated with the given ZIP code.
        - Filters the dataset to include only records from that city.
        - Returns daily price data for the city to support trend plotting.

    Args:
        zip_code (str): Five-digit ZIP code string used to identify the city.

    Returns:
        dict: JSON response containing:
            - data (list of dict): Daily price records with 'date' and 'price' keys.
            - city (str): Name of the city corresponding to the ZIP code.
    """

    df_one_city_frontend, city = get_df_one_city(house_TS_df, zip_code) # get the data frame to plot the trend for a metropolian area

    return {'data': df_one_city_frontend.to_dict('records'), 'city': city}


@app.get('/price_all_cities')
def price_all_cities():
    """
    Retrieve daily average housing prices grouped by city across all cities in the dataset.

    The endpoint performs the following steps:
        - Groups the dataset by 'city' and 'date'.
        - Calculates the mean 'price' for each city on each date.
        - Returns the aggregated data for comparison across cities.

    Returns:
        dict: JSON response containing:
            - data (list of dict): Records with 'city', 'date', and average 'price'.
    """

    df_all_cities_frontend = get_df_all_cities(house_TS_df) # get the dataframe to make a comparison over the US.
    return {'data': df_all_cities_frontend.to_dict('records')}
