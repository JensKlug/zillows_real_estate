# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import StandardScaler, RobustScaler
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler, RobustScaler
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.pipeline import Pipeline

# def preprocess_features(input_df: pd.DataFrame) -> np.ndarray:
#     """
#     Preprocess the input dataframe by scaling numerical features.
#     Drops 'zip_code' column if present.

#     Args:
#         input_df (pd.DataFrame): Raw input dataframe for prediction

#     Returns:
#         np.ndarray: Preprocessed and scaled feature array ready for model input
#     """
#     # Work on a copy to avoid side effects
#     df = input_df.copy()

#     # Drop unwanted columns if present
#     df = df.drop(columns=['zip_code'], errors='ignore')

#     # Define features (all columns are numeric features)
#     features = df.columns.tolist()

#     # Define numerical scaler transformer
#     preprocessor = ColumnTransformer(
#         transformers=[
#         ('std', StandardScaler(), ['latitude', 'longitude']),
#         ('rob', RobustScaler(), ['bed', 'bath', 'acre_lot', 'house_size', 'ppsf_zipcode'])
#         ],
#         remainder='drop'
#         )

#     # Fit and transform input dataframe (stateless for inference, so just transform)
#     X_processed = preprocessor.fit_transform(df)
#     print("✅ X_processed, with shape", X_processed.shape)

#     return X_processed

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler
import pandas as pd
import numpy as np

def get_preprocessor():
    """
    Define and return the preprocessor used to scale the input features.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('std', StandardScaler(), ['latitude', 'longitude']),
            ('rob', RobustScaler(), ['bed', 'bath', 'acre_lot', 'house_size', 'ppsf_zipcode'])
        ],
        remainder='drop'
    )
    return preprocessor


def preprocess_features(input_df: pd.DataFrame, preprocessor: ColumnTransformer, fit=False) -> np.ndarray:
    """
    Preprocess the input dataframe by scaling features using the provided preprocessor.

    Args:
        input_df (pd.DataFrame): Raw input dataframe.
        preprocessor (ColumnTransformer): Predefined column transformer.
        fit (bool): Whether to fit the transformer (True = fit_transform, False = transform only).

    Returns:
        np.ndarray: Scaled feature array ready for model input.
    """
    df = input_df.copy()
    df = df.drop(columns=['zip_code'], errors='ignore')

    if fit:
        X_processed = preprocessor.fit_transform(df)
    else:
        X_processed = preprocessor.transform(df)

    print("✅ X_processed, with shape", X_processed.shape)
    return X_processed
