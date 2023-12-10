import os
import pickle
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RFClassifier

from src.model.abc.classifier import Classifier


class RandomForestClassifier(Classifier):
    model_file = "simple_classifier.pkl"
    name = "Simple model"

    def __init__(self, n_estimators: int, class_weight: str, n_jobs: int, model_path: str, **kwargs):
        self.n_estimators = n_estimators
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.model = self._build_model(**kwargs)
        self.model_path = model_path

        self.features: Optional[List[str]] = None

    def _build_model(self, **kwargs) -> RFClassifier:
        return RFClassifier(n_estimators=self.n_estimators,
                            class_weight=self.class_weight,
                            n_jobs=self.n_jobs,
                            **kwargs)

    def get_feature_importance(self) -> Dict[str, float]:
        feature_importance = self.model.feature_importances_
        indices = np.argsort(feature_importance)[::-1]
        return {self.features[i]: feature_importance[i] for i in indices}

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        self.model.fit(X=X, y=y)
        self.features = X.columns

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X=X)

    def predict_proba(self, X: pd.DataFrame):
        return self.model.predict_proba(X=X)

    def save_model(self):
        with open(os.path.join(self.model_path, self.model_file), 'wb') as handler:
            pickle.dump(self.model, handler)
