#def initialize_model() #deeplearning
#def compile_model() #deeplearning
def train_model(X_train, y_train, param_grid=None, cv=5):
    # 6. Create pipeline with XGBoost
    pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(random_state=42, n_jobs=-1))
    ])

    # 7. Define hyperparameter grid for GridSearchCV
    param_grid = {
        'regressor__n_estimators': [100, 200],  # Number of trees
        'regressor__learning_rate': [0.01, 0.1],  # Step size for boosting
        'regressor__max_depth': [3, 5],  # Depth of trees
        'regressor__min_child_weight': [1, 3]  # Minimum sum of instance weight needed in a child
    }

    # 8. Perform GridSearchCV
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,  # 3-fold cross-validation
        scoring='neg_mean_squared_error',
        n_jobs=-1,  # Use all cores
        verbose=1
    )

    # 9. Fit the model
    grid_search.fit(X_train, y_train)


    # Log to MLflow
    best_model = grid_search.best_estimator_
    mlflow.log_params(grid_search.best_params_)
    # Note: Logging metrics (e.g., RMSE) moved to evaluate() - see feedback

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

# # Log to MLflow
#     best_model = grid_search.best_estimator_
#     mlflow.log_params(grid_search.best_params_)
#     y_pred = best_model.predict(X)
#     rmse = np.sqrt(mean_squared_error(y, y_pred))
#     mlflow.log_metrics({"rmse": rmse})
