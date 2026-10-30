"""FedPBA allocator: train deployable action predictors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from paper2.src.config import LEVELS
from paper2.src.predictors import encode_features

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:
    HAS_XGB = False


ALLOCATOR_MODELS = ["gradient_boosting", "random_forest", "logistic_regression"]
if HAS_XGB:
    ALLOCATOR_MODELS.append("xgboost")


def _build_model(name: str, seed: int = 42) -> Any:
    if name == "gradient_boosting":
        return HistGradientBoostingClassifier(max_iter=150, max_depth=6, random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=120, max_depth=10, random_state=seed, n_jobs=-1)
    if name == "logistic_regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1500, random_state=seed)),
            ]
        )
    if name == "xgboost" and HAS_XGB:
        return XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            objective="multi:softprob",
            num_class=len(LEVELS),
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
        )
    raise ValueError(f"Unknown allocator model: {name}")


@dataclass
class FedPBAAllocator:
    model_name: str
    feature_set: str
    feature_cols: List[str]
    model: Any
    label_encoder: LabelEncoder
    cat_encoder: Any
    cat_cols: List[str]

    def fit(self, df: pd.DataFrame, y_col: str = "oracle_action") -> "FedPBAAllocator":
        X, used, enc, cats = encode_features(df, self.feature_cols)
        self.feature_cols = used
        self.cat_encoder = enc
        self.cat_cols = cats
        le = LabelEncoder()
        le.fit(LEVELS)
        self.label_encoder = le
        self.model = _build_model(self.model_name)
        self.model.fit(X, le.transform(df[y_col].values))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X, _, _, _ = encode_features(df, self.feature_cols, self.cat_encoder, self.cat_cols)
        idx = self.model.predict(X)
        return self.label_encoder.inverse_transform(idx)

    def predict_proba(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        X, _, _, _ = encode_features(df, self.feature_cols, self.cat_encoder, self.cat_cols)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        inner = self.model.named_steps["clf"] if hasattr(self.model, "named_steps") else self.model
        if hasattr(inner, "predict_proba"):
            return inner.predict_proba(X)
        return None


def train_allocator(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    feature_set: str,
    model_name: str = "gradient_boosting",
) -> FedPBAAllocator:
    alloc = FedPBAAllocator(model_name, feature_set, feature_cols, None, None, None, [])
    return alloc.fit(train_df)
