from typing import Optional
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

_GRADE_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6}

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds engineered features to a pandas DataFrame:
    - dti = loan_amnt / person_income
    - grade_risk = mapped numeric score from loan_grade (A=1..F=6)
    """
    def __init__(self, drop_original_na: bool = False, clip_dti: Optional[float] = 10.0):
        self.drop_original_na = drop_original_na
        self.clip_dti = clip_dti

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        # dti
        with np.errstate(divide="ignore", invalid="ignore"):
            dti = df.get("loan_amnt", 0) / np.where(df.get("person_income", 1) == 0, np.nan, df.get("person_income", 1))
        if self.clip_dti is not None:
            dti = np.clip(dti, 0, self.clip_dti)
        df["dti"] = dti

        # grade risk
        grade = df.get("loan_grade")
        if grade is not None:
            df["grade_risk"] = grade.map(_GRADE_MAP).fillna(np.nan)
        else:
            df["grade_risk"] = np.nan

        if self.drop_original_na:
            df = df.dropna(axis=0, how="any")

        return df
