from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.hotel_booking_ml.features.feature_builder import FeatureBuilder

def make_preprocessor() -> ColumnTransformer:
    numeric_selector = make_column_selector(dtype_include=np.number)
    categorical_selector = make_column_selector(dtype_exclude=np.number)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=0.02,
                    sparse_output=True,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_selector),
            ("cat", categorical_pipeline, categorical_selector),
        ],
        remainder="drop",
    )

def make_modeling_pipeline(model) -> Pipeline:
    return Pipeline(
        steps=[
            ("features", FeatureBuilder()),
            ("preprocessor", make_preprocessor()),
            ("model", model),
        ]
    )
