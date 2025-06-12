
import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from zillow.ml_logic.model import train_model, evaluate_model
from zillow.ml_logic.preprocessor import preprocess_features
from zillow.ml_logic.registry import save_model, load_model
import mlflow
import joblib

# Adjust path to include ml_logic location (two levels up from zillow/api/)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Load raw data globally
rootpath = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
house_df = pd.read_csv(f'{rootpath}/raw_data/realtor-data.csv')  # Path relative to zillow/api/

def preprocess(house_df):
    """
    Preprocess the raw house data and save the cleaned dataset.
    """
    # Load and clean data
    cleaned_house_df, preprocessor = preprocess_features(house_df)

    output_path = os.path.join('../raw_data', f'cleaned_house_data_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv')
    if not os.path.exists('../raw_data'):
        os.makedirs('../raw_data')
    pd.DataFrame(cleaned_house_df, columns=['bed', 'bath', 'acre_lot', 'house_size', 'ppsf_zipcode', 'zip_code']).to_csv(output_path, index=False)
    print(f"✅ Preprocessed data saved to {output_path} with {len(cleaned_house_df)} rows")
    return cleaned_house_df

def train(cleaned_house_df):
    """
    Train the XGBoost model with GridSearchCV and log with MLflow.
    """
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("zillow_price_training")

    with mlflow.start_run():
        # Prepare data
        X = cleaned_house_df.drop(columns=['price'])
        y = cleaned_house_df['price']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model, history = train_model(X_train, y_train)

        # Save the best model
        model_path = 'model/xgboost_best_model.pkl'
        save_model(model, model_path)
        mlflow.log_artifact(model_path)
        print(f"✅ Model trained and saved to {model_path}")
        return model

def evaluate(cleaned_house_df,model):
    """
    Evaluate the trained model on the test dataset.
    """
    # Load the trained model
    model = load_model('model/xgboost_best_model.pkl')

    # Prepare data for evaluation
    X = cleaned_house_df.drop(columns=['price'])
    y = cleaned_house_df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    rmse = metrics['rmse']
    mae = metrics['mae']
    r2 = metrics['r2']
    print(f"✅ Evaluation - RMSE: ${rmse:,.2f}, MAE: ${mae:,.2f}, R²: {r2:.4f}")
    return metrics

def pred(input_data):
    """
    Make a prediction for a single house input.
    """
    # Load the trained model
    model = load_model('model/xgboost_best_model.pkl')

    # Prepare input data as a DataFrame
    columns = ['bed', 'bath', 'acre_lot', 'house_size', 'ppsf_zipcode', 'zip_code']
    input_df = pd.DataFrame([input_data], columns=columns)

    # Predict
    prediction = model.predict(input_df)[0]
    print(f"✅ Prediction for input: ${prediction:,.2f}")
    return {"predicted_price": float(prediction)}

if __name__ == '__main__':
    # Ensure raw_data directory exists
    if not os.path.exists('../raw_data'):
        os.makedirs('../raw_data')

    # Execute the workflow
    cleaned_house_df = preprocess(house_df)
    model = train(cleaned_house_df)
    metrics = evaluate(cleaned_house_df,model)
    prediction = pred(X_pred)
