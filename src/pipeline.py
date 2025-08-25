from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from .features import FeatureEngineer
from .config import CATEGORICAL_COLS, NUMERIC_COLS

def build_preprocessing_pipeline():
    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical, CATEGORICAL_COLS),
            ("num", numeric, NUMERIC_COLS),
        ],
        remainder="drop"
    )

    full_pipe = Pipeline(steps=[
        ("feat", FeatureEngineer()),
        ("prep", preprocessor),
    ])
    return full_pipe
