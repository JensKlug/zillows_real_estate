import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ml_logic.model import train_model, evaluate_model
from ml_logic.preprocessor import preprocess_features
from ml_logic.registry import save_model, load_model
import mlflow
import joblib
import os

# Load raw data globally
house_df = pd.read_csv('../raw_data/realtor-data.csv')  # Adjust path as needed

def preprocess():
    """
    Preprocess the raw house data and save the cleaned dataset.
    """
    # Load and clean data
    cleaned_house_df = preprocess_features(house_df)  # Needs adjustment, see below
    output_path = os.path.join('../raw_data', f'cleaned_house_data_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv')
    if not os.path.exists('../raw_data'):
        os.makedirs('../raw_data')
    pd.DataFrame(cleaned_house_df, columns=['bed', 'bath', 'acre_lot', 'house_size', 'ppsf_zipcode', 'zip_code']).to_csv(output_path, index=False)
    print(f"✅ Preprocessed data saved to {output_path} with {len(cleaned_house_df)} rows")
    return cleaned_house_df

def train():
    """
    Train the XGBoost model with GridSearchCV and log with MLflow.
    """
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("zillow_price_training")

    with mlflow.start_run():
        # Prepare data
        X = cleaned_house_df.drop(columns=['price'])
        y = cleaned_house_df['price']

        # Split data (since model.py expects X_train, y_train)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model, history = train_model(X_train, y_train)  # Pass training data

        # Save the best model
        model_path = 'model/xgboost_best_model.pkl'
        save_model(model, model_path)
        mlflow.log_artifact(model_path)
        print(f"✅ Model trained and saved to {model_path}")
        return model

def evaluate():
    """
    Evaluate the trained model on the full dataset.
    """
    # Load the trained model
    model = load_model('model/xgboost_best_model.pkl')

    # Prepare data for evaluation
    X = cleaned_house_df.drop(columns=['price'])
    y = cleaned_house_df['price']

    # Evaluate
    metrics = evaluate_model(model, X, y)
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
    cleaned_house_df = preprocess()
    model = train()
    metrics = evaluate()
    sample_input = {"bed": 3, "bath": 2, "acre_lot": 0.5, "house_size": 2000, "ppsf_zipcode": 300, "zip_code": "90210"}
    prediction = pred(sample_input)
