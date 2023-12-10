import logging
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataSplit:
    def __init__(self, test_size: float, random_state: int, features: List[str], label: str):
        self.test_size = test_size
        self.random_state = random_state
        self.features = features
        self.label = label

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        X = df[self.features]
        y = df[self.label]
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
