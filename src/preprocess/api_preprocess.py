from typing import List, Dict

import pandas as pd


class ApiPreprocess:
    def __init__(self, features: List[str], label: str, map_neighb: Dict[str, int], map_room_type: Dict[str, int]):
        self.features = features
        self.label = label
        self.map_neighb = map_neighb
        self.map_room_type = map_room_type

    def preprocess(self, df: pd.DataFrame, with_label: bool) -> pd.DataFrame:
        columns = self.features.copy() if not with_label else self.features.copy() + [self.label]
        df = df[columns]
        df = self._map_to_categories(df)
        return df.dropna(axis=0)

    def _map_to_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        df["neighbourhood"] = df["neighbourhood"].map(self.map_neighb)
        df["room_type"] = df["room_type"].map(self.map_room_type)
        return df
