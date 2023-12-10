from abc import ABC, abstractmethod
from typing import Union, List

import numpy as np
import pandas as pd


ml_io = Union[pd.DataFrame, np.ndarray, List[float]]


class Classifier(ABC):
    name: str = NotImplemented

    @abstractmethod
    def fit(self, X: ml_io, y: ml_io):
        return NotImplemented

    @abstractmethod
    def get_feature_importance(self):
        return NotImplemented

    @abstractmethod
    def predict(self, X: ml_io) -> ml_io:
        return NotImplemented

    @abstractmethod
    def predict_proba(self, X: ml_io) -> ml_io:
        return NotImplemented
