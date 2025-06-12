def preprocess_features():
    def create_sklearn_preprocessor():
        # 4. Preprocess with ColumnTransformer (using RobustScaler)
        preprocessor = ColumnTransformer(
        transformers=[
        ('std', StandardScaler(), ['latitude', 'longitude']),
        ('rob', RobustScaler(), ['bed', 'bath', 'acre_lot', 'house_size', 'ppsf_zipcode'])
        ],
        remainder='passthrough'
        )
        return final_preprocessor

    preprocessor = create_sklearn_preprocessor()
    X_processed = preprocessor.fit_transform(X)

    print("✅ X_processed, with shape", X_processed.shape)

    return X_processed
