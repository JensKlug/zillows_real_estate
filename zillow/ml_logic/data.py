import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
import pgeocode
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import joblib
import os
from datetime import datetime


def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))  # .../zillow/ml_logic
    raw_data_dir = os.path.abspath(os.path.join(base_dir, '..', '..', 'raw_data'))

    area_path = os.path.join(raw_data_dir, 'HouseTS.csv')
    house_path = os.path.join(raw_data_dir, 'realtor-data.csv')

    area_df = pd.read_csv(area_path)
    house_df = pd.read_csv(house_path)

    unique_zipcodes_area_df = area_df['zipcode'].unique().tolist()
    house_df = house_df[house_df['zip_code'].isin(unique_zipcodes_area_df)]

    return house_df


def clean_data(df):
    # Drop columns 'brokered_by', 'status'
    df = df.drop(columns=['brokered_by', 'status'])

     # Drop duplicates
    df = df.drop_duplicates()

    # Drop columns 'street', 'city', 'state' and 'prev_sold_date'
    df = df.drop(columns=['street', 'city', 'state', 'prev_sold_date'])

    # Drop rows with NaN values from 'price'
    df = df.dropna(subset=['price'])

    # Create list where 'bed' & 'bath' & 'house_size' are NaN
    nan_values = df[
        (pd.isna(df['bed'])) &
        (pd.isna(df['bath'])) &
        (pd.isna(df['house_size']))
    ]

    # Filter out rows that are in nan_values because we assume they are land sales
    df = df[~df.index.isin(nan_values.index)]

    # Impute missing data
    df['bed'] = df['bed'].fillna(df['bed'].median())
    df['bath'] = df['bath'].fillna(df['bath'].median())
    df['house_size'] = df['house_size'].fillna(df['house_size'].median())
    df['acre_lot'] = df['acre_lot'].fillna(0)

    # Calculate PPSF for each row
    df['ppsf'] = round(df['price'] / df['house_size'], 2)

    # Calculate median PPSF per zip_code
    ppsf_median = df.groupby('zip_code')['ppsf'].median().reset_index(name='ppsf_zipcode')

    # Merge median PPSF back to df
    df = df.merge(ppsf_median, on='zip_code', how='left')

    # Convert zipcode to int
    df['zip_code'] = df['zip_code'].astype(int)

    # Drop temporary ppsf column
    df = df.drop(columns=['ppsf'])

    # Calculate boundaries for 'price', 'acre_lot', 'house_size', 'ppsf_zipcode'
    lower_price = df['price'].quantile(0.03)
    upper_price = df['price'].quantile(0.97)
    upper_house_size = df['house_size'].quantile(0.99)
    lower_acre_lot = df['acre_lot'].quantile(0.01)
    upper_acre_lot = df['acre_lot'].quantile(0.99)
    lower_ppsf_zipcode = df['ppsf_zipcode'].quantile(0.03)
    upper_ppsf_zipcode = df['ppsf_zipcode'].quantile(0.97)

    # Apply boundaries to df
    df = df[
        (df['price'] > lower_price) &
        (df['price'] < upper_price) &
        (df['bed'] < 14) &
        (df['bath'] < 12) &
        (df['house_size'] < upper_house_size) &
        (df['acre_lot'] > lower_acre_lot) &
        (df['acre_lot'] < upper_acre_lot) &
        (df['ppsf_zipcode'] > lower_ppsf_zipcode) &
        (df['ppsf_zipcode'] < upper_ppsf_zipcode)
        ]

    return df

def convert_zipcode(df):
    # Convert zip_code column to 5-digit string
    df['zip_code'] = df['zip_code'].astype(str).str.replace('\.0$', '', regex=True).str.zfill(5)

    # Get unique zip codes
    unique_zips = df['zip_code'].unique()

    # Initialize pgeocode for US
    nomi = pgeocode.Nominatim('us')

    # Function to get coordinates
    def get_coordinates(zip_code):
        try:
            result = nomi.query_postal_code(zip_code)
            if result.empty or pd.isna(result.latitude):
                return pd.Series([None, None])
            return pd.Series([result.latitude, result.longitude])
        except:
            return pd.Series([None, None])

    # Create DataFrame for unique zip codes
    zip_coords = pd.DataFrame(unique_zips, columns=['zip_code'])
    zip_coords[['latitude', 'longitude']] = zip_coords.apply(lambda row: get_coordinates(row['zip_code']), axis=1)

    # Map coordinates back to filtered_house_df
    coords_dict = zip_coords.set_index('zip_code')[['latitude', 'longitude']].to_dict('index')
    df['latitude'] = df['zip_code'].map(lambda x: coords_dict.get(x, {}).get('latitude'))
    df['longitude'] = df['zip_code'].map(lambda x: coords_dict.get(x, {}).get('longitude'))

    # Drop 'zip_code' column
    df = df.drop(columns=['zip_code'])

    return df
def create_zip_dict(df):
    """
    Create a dictionary mapping zip codes to a single ppsf_zipcode value.
    Example: {12345: 210.5, 67890: 198.3}
    """
    zip_dict = (
        df.drop_duplicates(subset="zip_code")
          .set_index("zip_code")[["ppsf_zipcode"]]
          .to_dict(orient="index")
    )

    zip_dict = {
        zip_code: values["ppsf_zipcode"]  # remove the [ ] list wrapping
        for zip_code, values in zip_dict.items()
    }

    return zip_dict

def prepare_user_input(user_input: dict, zip_dict: dict) -> pd.DataFrame:
    """
    user_input: e.g. {'bed': 3, 'bath': 2, 'acre_lot': 0.25, 'zip_code': 12345, 'house_size': 1800}
    ppsf_zip_dict: {12345: [210.5], 67890: [198.3], ...}
    default_ppsf: fallback value if zip_code not found
    """
    ppsf = zip_dict.get(user_input['zip_code'])

    # Build dataframe for model input, replacing zip_code by ppsf_zipcode
    data = {
        'bed': user_input['bed'],
        'bath': user_input['bath'],
        'acre_lot': user_input['acre_lot'],
        'house_size': user_input['house_size'],
        'ppsf_zipcode': ppsf,
    }

    return pd.DataFrame([data])
