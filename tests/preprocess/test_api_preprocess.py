from unittest import TestCase

import pytest

from src.preprocess.api_preprocess import ApiPreprocess


@pytest.mark.usefixtures("get_config", "get_processed_data")
class TestApiPreprocess(TestCase):
    def test_api_preprocess(self):
        input_df = self.processed_df

        train_preprocess = ApiPreprocess(**self.config["api_preprocess"])
        output_df = train_preprocess.preprocess(input_df, with_label=True)

        self.assertSetEqual(set(output_df["neighbourhood"].unique()),
                            set(self.config["api_preprocess"]["map_neighb"].values()))
        self.assertSetEqual(set(output_df["room_type"].unique()),
                            set(self.config["api_preprocess"]["map_room_type"].values()))
