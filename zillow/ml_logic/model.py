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
    """
    Train an XGBoost regression model with hyperparameter tuning using GridSearchCV.

    The function performs the following steps:
        - Initializes an XGBoost regressor with a fixed random state.
        - Defines a default hyperparameter grid (if not provided).
        - Performs grid search with cross-validation to identify the best model.
        - Prints the number of training samples, best hyperparameters, and cross-validated RMSE.

    Args:
        X_train (pd.DataFrame or np.ndarray): Feature matrix for training.
        y_train (pd.Series or np.ndarray): Target values for training.
        param_grid (dict, optional): Dictionary specifying hyperparameter ranges to search over.
            If None, a default grid is used with parameters:
                - 'n_estimators': [100, 200]
                - 'learning_rate': [0.01, 0.1]
                - 'max_depth': [3, 5]
                - 'min_child_weight': [1, 3]
        cv (int, optional): Number of cross-validation folds. Defaults to 5.

    Returns:
        XGBRegressor: Trained XGBoost regressor with the best-found hyperparameters.
    """

    # Dfine the model XGBoost
    regressor = XGBRegressor(random_state=42, n_jobs=-1)

    # Define hyperparameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [100, 200],  # Number of trees
        'learning_rate': [0.01, 0.1],  # Step size for boosting
        'max_depth': [3, 5],  # Depth of trees
        'min_child_weight': [1, 3]  # Minimum sum of instance weight needed in a child
    }

    # Perform GridSearchCV
    grid_search = GridSearchCV(
        regressor,
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,  # Use all cores
        verbose=1
    )

    # Fit the model
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    print(f"✅ Model trained on {len(X_train)} rows with best parameters: {grid_search.best_params_}")
    print(f"Min cross-validated RMSE: ${np.sqrt(-grid_search.best_score_):,.2f}")

    return best_model, grid_search

def evaluate_model(model, X, y, batch_size=64):
    """
    Evaluate the performance of a trained XGBoost regression model on a given dataset.

    The function performs the following steps:
        - Predicts target values using the provided model and feature matrix.
        - Computes evaluation metrics: Root Mean Squared Error (RMSE),
          Mean Absolute Error (MAE), and R² score.
        - Prints and returns the evaluation metrics.

    Note:
        The `batch_size` argument is included for compatibility but is not used,
        as XGBoost processes all rows at once during prediction.

    Args:
        model (XGBRegressor): Trained XGBoost regression model.
        X (pd.DataFrame or np.ndarray): Feature matrix for evaluation.
        y (pd.Series or np.ndarray): True target values.
        batch_size (int, optional): Placeholder for batch processing (not used). Defaults to 64.

    Returns:
        dict or None: Dictionary containing the evaluation metrics:
            {
                'rmse': float,
                'mae': float,
                'r2': float
            }
            Returns None if no model is provided.
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
