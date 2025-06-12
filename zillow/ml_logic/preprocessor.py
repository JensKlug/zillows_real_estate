from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def preprocess_features(input_df: pd.DataFrame) -> np.ndarray:
    """
    Preprocess the input dataframe by scaling numerical features.
    Drops 'zip_code' column if present.

    Args:
        input_df (pd.DataFrame): Raw input dataframe for prediction

    Returns:
        np.ndarray: Preprocessed and scaled feature array ready for model input
    """
    # Work on a copy to avoid side effects
    df = input_df.copy()

    # Drop unwanted columns if present
    df = df.drop(columns=['zip_code'], errors='ignore')

    # Define features (all columns are numeric features)
    features = df.columns.tolist()

    # Define numerical scaler transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), features)
        ],
        remainder='drop'
    )

    # Fit and transform input dataframe (stateless for inference, so just transform)
    X_processed = preprocessor.fit_transform(df)

    return X_processed
