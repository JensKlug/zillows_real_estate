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
from sklearn.impute import SimpleImputer

def load_data():
    # Always get the directory of the *current script file*
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Go up two levels to reach the root project directory
    project_root = os.path.abspath(os.path.join(base_dir, '..', '..'))
    raw_data_dir = os.path.join(project_root, 'raw_data')

    area_path = os.path.join(raw_data_dir, 'HouseTS.csv')
    house_path = os.path.join(raw_data_dir, 'realtor-data.csv')

    area_df = pd.read_csv(area_path)
    house_df = pd.read_csv(house_path)

    #check initial row count
    print(f"Initial rows in house_df: {len(house_df)}")

    # Filter house_df by available zipcodes in area_df
    unique_zipcodes_area_df = area_df['zipcode'].unique().tolist()
    house_df = house_df[house_df['zip_code'].isin(unique_zipcodes_area_df)]

    #check row count after filtering
    print(f"Rows after zip code filter: {len(house_df)}")

    return house_df



def clean_data(df):
    """
    Clean and preprocess the input DataFrame for model training or prediction.

    The function performs the following steps:
        - Drops unnecessary columns: 'brokered_by', 'status', 'street', 'city', 'state', 'prev_sold_date'.
        - Removes duplicate rows and rows with missing 'price'.
        - Removes likely land sales where 'bed', 'bath', and 'house_size' are all NaN.
        - Applies median imputation for 'bed', 'bath', and 'house_size'.
        - Applies constant imputation (0) for 'acre_lot'.
        - Computes price per square foot (PPSF) and adds the zip-code-level median as 'ppsf_zipcode'.
        - Adds geographic features ('latitude', 'longitude') via the `convert_zipcode` function.
        - Removes rows with missing latitude/longitude.
        - Filters out outliers based on quantile thresholds for 'price', 'house_size', 'acre_lot', and 'ppsf_zipcode'.
        - Filters out unrealistic values for 'bed' and 'bath'.

    Args:
        df (pd.DataFrame): Raw real estate data with features including
            ['price', 'bed', 'bath', 'house_size', 'acre_lot', 'zip_code',
            'brokered_by', 'status', 'street', 'city', 'state', 'prev_sold_date'].

    Returns:
        pd.DataFrame: Cleaned and filtered DataFrame ready for modeling.
    """

    print(f"Initial rows in clean_data: {len(df)}")
    # Drop columns 'brokered_by', 'status'
    df = df.drop(columns=['brokered_by', 'status'])

     # Drop duplicates
    df = df.drop_duplicates()
    print(f"Rows after dropping duplicates: {len(df)}")

    # Drop columns 'street', 'city', 'state' and 'prev_sold_date'
    df = df.drop(columns=['street', 'city', 'state', 'prev_sold_date'])

    # Drop rows with NaN values from 'price'
    df = df.dropna(subset=['price'])
    print(f"Rows after dropping NaN prices: {len(df)}")

    # Create list where 'bed' & 'bath' & 'house_size' are NaN
    nan_values = df[
        (pd.isna(df['bed'])) &
        (pd.isna(df['bath'])) &
        (pd.isna(df['house_size']))
    ]
    print(f"Rows after dropping land sales (NaN bed, bath, house_size): {len(df)}")

    # Filter out rows that are in nan_values because we assume they are land sales
    df = df[~df.index.isin(nan_values.index)]

    # Define imputers for different strategies
    imputer_median = SimpleImputer(strategy='median')
    imputer_constant = SimpleImputer(strategy='constant', fill_value=0)
    # Apply imputation to the respective columns
    df[['bed', 'bath', 'house_size']] = imputer_median.fit_transform(df[['bed', 'bath', 'house_size']])
    df['acre_lot'] = imputer_constant.fit_transform(df[['acre_lot']]).ravel()

    # Calculate PPSF
    df['ppsf'] = np.where(df['house_size'] > 0, round(df['price'] / df['house_size'], 2), 0)  # Avoid div by zero
    ppsf_median = df.groupby('zip_code')['ppsf'].median().reset_index(name='ppsf_zipcode')
    df = df.merge(ppsf_median, on='zip_code', how='left')

    # Convert zipcode into longitude and latitude
    df = convert_zipcode(df)
    df = df.dropna(subset=['latitude', 'longitude'])
    print(f"Rows after dropping NaN latitude/longitude: {len(df)}")

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
    print(f"Rows after outlier filtering: {len(df)}")

    return df


def create_zip_dict(df):
    """
    Create a dictionary mapping zip codes to a dict of 'ppsf_zipcode', 'longitude', and 'latitude'.
    Example:
        {
            12345: {"ppsf_zipcode": 210.5, "longitude": -73.99, "latitude": 40.75},
            67890: {"ppsf_zipcode": 198.3, "longitude": -74.01, "latitude": 40.78}
        }
    """
    # Select the needed columns and drop duplicates by zip_code
    zip_df = df.drop_duplicates(subset="zip_code")[["zip_code", "ppsf_zipcode", "longitude", "latitude"]]

    # Set zip_code as index and convert to dict of dicts
    zip_dict = zip_df.set_index("zip_code").to_dict(orient="index")

    return zip_dict

def prepare_user_input(user_input: dict, zip_dict: dict) -> pd.DataFrame:
    """
    Prepare user input dict into a DataFrame for prediction,
    replacing 'zip_code' with 'ppsf_zipcode', 'longitude', and 'latitude' from zip_dict.

    Args:
        user_input: Dictionary with keys like 'bed', 'bath', 'acre_lot', 'zip_code', 'house_size'.
        zip_dict: Dictionary mapping zip_code to dict of {'ppsf_zipcode', 'longitude', 'latitude'}.

    Returns:
        pd.DataFrame: Single-row DataFrame ready for model input.
    """
    zip_info = zip_dict.get(user_input['zip_code'])

    data = {
    'latitude': zip_info.get('latitude'),
    'longitude': zip_info.get('longitude'),
    'bed': user_input['bed'],
    'bath': user_input['bath'],
    'acre_lot': user_input['acre_lot'],
    'house_size': user_input['house_size'],
    'ppsf_zipcode': zip_info.get('ppsf_zipcode'),
    }

    return pd.DataFrame([data])


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

    return df


def get_df_one_city(house_TS_df, zipcode):
    house_TS_df['zipcode'] = house_TS_df['zipcode'].astype(str).str.zfill(5)
    city = house_TS_df[house_TS_df['zipcode'] == zipcode].city.values[0]# City with upto 3 letters as a string.
    df_one_city = house_TS_df[house_TS_df['city'] == city][['date','price']]

    return df_one_city, city # might have a size of ~ 675 KB -> so every time less than 1 MB.


def get_df_all_cities(house_TS_df):
    df_grouped_by_city_date_mean = house_TS_df.groupby(['city', 'date'])['price'].mean().reset_index()

    return df_grouped_by_city_date_mean # This has a size of memory usage: 100.0+ KB.

def get_df_yearly_data(house_TS_df, zipcode):

    house_TS_df['date'] = pd.to_datetime(house_TS_df['date'])  # if not already
    house_TS_df['year'] = house_TS_df['date'].dt.year

    dfprice_yearly = house_TS_df.groupby(['city', 'zipcode', 'year'])[
    ['median_sale_price','Median Home Value']].mean().reset_index()
    dfprice_yearly['zipcode'] = dfprice_yearly['zipcode'].astype(str).str.zfill(5)
    zipcode_price_evolution = dfprice_yearly[dfprice_yearly['zipcode'] == zipcode]

    return zipcode_price_evolution #528.0+ bytes



def df_for_zipcode_graph(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares the house time series DataFrame for ZIP code graphing:
    - Converts 'zipcode' to 5-character strings (zero-padded)
    - Parses 'date' column as datetime
    """
    df = df.copy()
    df['zipcode'] = df['zipcode'].astype(str).str.zfill(5)
    df['date'] = pd.to_datetime(df['date'])
    return df
