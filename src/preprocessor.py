import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data import bool_features, cat_features, num_features


class NumericToBoolTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Convert to a boolean array: True if value is not NaN, False if NaN
        return ~np.isnan(X)


def numeric_to_bool():
    return ("numeric_to_bool", NumericToBoolTransformer())


def onehot():
    return ("onehot", OneHotEncoder(handle_unknown="error", sparse_output=False))


def imputation_constant(constant=None):
    fill_value = constant if constant is not None else "missing"
    return (
        "imputation_constant",
        SimpleImputer(fill_value=fill_value, strategy="constant")
    )


def scalar():
    return ("scaler", StandardScaler())


def to_categorical():
    return ("to_categorical", FunctionTransformer(lambda X: X.astype('object')))


def standard_cat_pipeline():
    return ("categorical", Pipeline(
        steps=[
            imputation_constant(),
            onehot(),
        ]
    ), cat_features)


def standard_bool_pipeline():
    return ("boolean", Pipeline(
        steps=[
            to_categorical(),
            onehot(),
        ]
    ), bool_features)


def standard_num_pipeline():
    return ("numerical", Pipeline(
        steps=[
            imputation_constant(-1),
            scalar(),
        ]
    ), num_features)


def make_preprocessor(transformers):
    return ("preprocessor", ColumnTransformer(
        transformers=transformers
    ))


def standard_preprocessor():
    return make_preprocessor([
        standard_cat_pipeline(),
        standard_bool_pipeline(),
        standard_num_pipeline(),
    ])
