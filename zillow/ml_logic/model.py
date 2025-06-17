import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import mlflow
import joblib
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from zillow.ml_logic.data import load_data, clean_data,convert_zipcode

def train_model(X_train, y_train, param_grid=None, cv=5):

    # 6.define the model XGBoost

    regressor = XGBRegressor(random_state=42, n_jobs=-1)


    # 7. Define hyperparameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [100, 200],  # Number of trees
        'learning_rate': [0.01, 0.1],  # Step size for boosting
        'max_depth': [3, 5],  # Depth of trees
        'min_child_weight': [1, 3]  # Minimum sum of instance weight needed in a child
    }

    # 8. Perform GridSearchCV
    grid_search = GridSearchCV(
        regressor,
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,  # Use all cores
        verbose=1
    )

    # 9. Fit the model
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    print(f"✅ Model trained on {len(X_train)} rows with best parameters: {grid_search.best_params_}")
    print(f"Min cross-validated RMSE: ${np.sqrt(-grid_search.best_score_):,.2f}")

    return best_model, grid_search
def evaluate_model(model, X, y, batch_size=64):
    """
    Evaluate trained XGBoost model performance on the dataset.
    """
    print(f"\nEvaluating model on {len(X)} rows...")

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    # Predict (no batch_size in XGBoost, processes all at once)
    y_pred = model.predict(X)

    # Compute metrics
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"✅ Model evaluated, RMSE: ${rmse:,.2f}, MAE: ${mae:,.2f}, R²: {r2:.4f}")

    return {"rmse": rmse, "mae": mae, "r2": r2}
