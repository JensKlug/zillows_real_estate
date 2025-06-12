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

    # Step 2: Calculate PPSF for each row
    df['ppsf'] = round(df['price'] / df['house_size'], 2)

    # Step 3: Calculate median PPSF per zip_code
    ppsf_median = df.groupby('zip_code')['ppsf'].median().reset_index(name='ppsf_zipcode')

    # Step 4: Merge median PPSF back to df
    df = df.merge(ppsf_median, on='zip_code', how='left')

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


def create_zip_dict(df):
    # Create the dictionary from the DataFrame
    zip_dict = df.drop_duplicates(subset="zip_code").set_index("zip_code")[["p_c_income", "ppsf_zipcode"]].to_dict(orient="index")

    # Convert inner dicts to lists
    zip_dict = {zip_code: [values["p_c_income"], values["ppsf_zipcode"]] for zip_code, values in zip_dict.items()}

    return zip_dict

# # Read HouseTS.csv into area_df
# area_df = pd.read_csv('../raw_data/HouseTS.csv')

# # Read realtor-data.csv into house_df
# house_df = pd.read_csv('../raw_data/realtor-data.csv')

# # Create list of unique zipcodes in area_df
# unique_zipcodes_area_df = area_df['zipcode'].unique().tolist()

# # Filter house_df by unique_zipcoes_area_df
# house_df = house_df[house_df['zip_code'].isin(unique_zipcodes_area_df)]

