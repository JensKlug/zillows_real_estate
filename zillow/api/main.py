
# import sys
# import os
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from zillow.ml_logic.model import train_model, evaluate_model
# from zillow.ml_logic.preprocessor import preprocess_features
# from zillow.ml_logic.registry import save_model, load_model
# import mlflow
# import joblib
# from sklearn.preprocessing import StandardScaler, RobustScaler
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.pipeline import Pipeline
# from zillow.ml_logic.data import load_data, clean_data,convert_zipcode
# from zillow.ml_logic.preprocessor import preprocess_features, get_preprocessor


# # Adjust path to include ml_logic location (two levels up from zillow/api/)
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# def preprocess(house_df):
#     """
#     Preprocess the raw house data with scaling and save the cleaned dataset.
#     """
#     # Scale data with preprocess_features
#     cleaned_house_df = preprocess_features(house_df)

#     # Define output path with timestamp
#     output_path = os.path.join('../raw_data', f'cleaned_house_data_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv')
#     if not os.path.exists('../raw_data'):
#         os.makedirs('../raw_data')

#     # Reconstruct DataFrame with all scaled columns
#     columns = ['latitude', 'longitude', 'bed', 'bath', 'acre_lot', 'house_size', 'ppsf_zipcode']
#     cleaned_house_df = pd.DataFrame(cleaned_house_df, columns=columns)
#     cleaned_house_df['price'] = house_df['price']  # Reattach price

#     # Save to CSV
#     cleaned_house_df.to_csv(output_path, index=False)
#     print(f"✅ Preprocessed data saved to {output_path} with {len(cleaned_house_df)} rows")
#     return cleaned_house_df



# def train(cleaned_house_df):
#     """
#     Train the XGBoost model with GridSearchCV and log with MLflow.
#     """

#     # Prepare data
#     X = cleaned_house_df.drop(columns=['price'])
#     y = cleaned_house_df['price']

#     print(f"y stats before split: NaN = {y.isna().sum()}, inf = {np.isinf(y).sum()}, max = {y.max()}, min = {y.min()}")


#     mask = ~y.isna()
#     X = X[mask]
#     y = y[mask]

#     print(f"y stats after NaN removal: NaN = {y.isna().sum()}, inf = {np.isinf(y).sum()}, max = {y.max()}, min = {y.min()}")


#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     print(f"y_train stats: NaN = {y_train.isna().sum()}, inf = {np.isinf(y_train).sum()}, max = {y_train.max()}, min = {y_train.min()}")

#     # Train model
#     model, history = train_model(X_train, y_train)

#     # Save the best model
#     model_path = save_model(model)
#     print(f"✅ Model trained and saved to {model_path}")
#     return model, X_test, y_test

# def evaluate(X_test,y_test,model):
#     """
#     Evaluate the trained model on the test dataset.
#     """
#     # Evaluate
#     metrics = evaluate_model(model, X_test, y_test)
#     if metrics is None:
#         print("❌ Evaluation failed, no metrics returned")
#         return None
#     rmse = metrics['rmse']
#     mae = metrics['mae']
#     r2 = metrics['r2']
#     print(f"✅ Evaluation - RMSE: ${rmse:,.2f}, MAE: ${mae:,.2f}, R²: {r2:.4f}")
#     return metrics

# def make_prediction(input_data):
#     """
#     Make a prediction for a single house input.
#     """
#     # Load the trained model
#     model = load_model()

#     # Prepare input data as a DataFrame
#     columns = ['latitude', 'longitude', 'bed', 'bath', 'acre_lot', 'house_size', 'ppsf_zipcode']
#     input_df = pd.DataFrame([input_data], columns=columns)

#     # Predict
#     prediction = model.predict(input_df)[0]
#     print(f"✅ Prediction for input: ${prediction:,.2f}")
#     return {"predicted_price": float(prediction)}

# if __name__ == '__main__':
#     # Ensure raw_data directory exists
#     if not os.path.exists('../raw_data'):
#         os.makedirs('../raw_data')

#     # Execute the workflow
#     house_df = load_data()
#     house_df = clean_data(house_df)
#     #house_df = convert_zipcode(house_df)
#     cleaned_house_df = preprocess(house_df)
#     model, X_test, y_test = train(cleaned_house_df)
#     metrics = evaluate(X_test,y_test,model)

#     sample_input = {
#     "latitude": 34.0522,
#     "longitude": -118.2437,
#     "bed": 3,
#     "bath": 2,
#     "acre_lot": 0.5,
#     "house_size": 2000,
#     "ppsf_zipcode": 300,
#     }
#     prediction = make_prediction(sample_input)


import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
from zillow.ml_logic.preprocessor import get_preprocessor, preprocess_features
from zillow.ml_logic.model import train_model, evaluate_model
from zillow.ml_logic.preprocessor import preprocess_features, get_preprocessor
from zillow.ml_logic.registry import save_model, load_model
from zillow.ml_logic.data import load_data, clean_data
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler
from zillow.ml_logic.registry import save_model


def preprocess(house_df):
    """
    Split and preprocess the raw house data with scaling.
    """
    X = house_df.drop(columns=['price'])
    y = house_df['price']

    mask = ~y.isna()
    X = X[mask]
    y = y[mask]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = get_preprocessor()
    X_train_processed = preprocess_features(X_train, preprocessor, fit=True)
    X_test_processed = preprocess_features(X_test, preprocessor, fit=False)

    joblib.dump(preprocessor, "models/preprocessor.pkl")
    print("✅ Preprocessor saved to models/preprocessor.pkl")

    return X_train_processed, X_test_processed, y_train, y_test

def train(X_train_processed, y_train):
    """
    Train the XGBoost model on raw price.
    """
    model, history = train_model(X_train_processed, y_train)
    print(f"✅ Model trained (target = raw price)")
    return model

def evaluate(X_test_processed, y_test, model):
    """
    Evaluate the trained model on test data using raw price metrics.
    """
    y_pred = model.predict(X_test_processed)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"✅ Evaluation - RMSE: ${rmse:,.2f}, MAE: ${mae:,.2f}, R²: {r2:.4f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def make_prediction(input_data, house_df, model):
    """
    Make a price prediction from input features using raw target model.
    """
    preprocessor = get_preprocessor()
    preprocessor.fit(house_df.drop(columns=['price'], errors='ignore'))

    input_df = pd.DataFrame([input_data])
    input_processed = preprocess_features(input_df, preprocessor, fit=False)

    prediction = model.predict(input_processed)[0]

    print(f"✅ Prediction for input: ${prediction:,.2f}")
    return {"predicted_price": float(prediction)}


if __name__ == '__main__':
    # Load and clean raw data
    house_df = load_data()
    house_df = clean_data(house_df)

    # Preprocess and split data
    X_train_processed, X_test_processed, y_train, y_test = preprocess(house_df)

    # Train model
    model = train(X_train_processed, y_train)

    model_path = save_model(model)

    # Evaluate
    metrics = evaluate(X_test_processed, y_test, model)

    # Sample prediction
    sample_input = {
        "latitude": 34.0522,
        "longitude": -118.2437,
        "bed": 3,
        "bath": 2,
        "acre_lot": 0.5,
        "house_size": 2000,
        "ppsf_zipcode": 300,
    }

    prediction = make_prediction(sample_input, house_df, model)
