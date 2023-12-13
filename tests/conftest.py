import pandas as pd
import pytest

from src.data.utils import load_config


@pytest.fixture(scope="class")
def get_config(request):
    request.cls.config = load_config("./tests/resources/config.yaml")


@pytest.fixture(scope="class")
def get_raw_data(request):
    request.cls.raw_df = pd.read_csv("./tests/resources/data/raw/listings.csv").sort_values("id").reset_index(drop=True)


@pytest.fixture(scope="class")
def get_processed_data(request):
    df = pd.read_csv("./tests/resources/data/processed/preprocessed_listings.csv")
    df = df.drop(columns="Unnamed: 0").sort_values("id").reset_index(drop=True)
    categories = pd.CategoricalDtype(categories=[0, 1, 2, 3], ordered=True)
    df["category"] = df["category"].astype(categories)
    request.cls.processed_df = df
