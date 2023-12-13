from unittest import TestCase

import pytest

from src.preprocess.train_preprocess import TrainPreprocess


@pytest.mark.usefixtures("get_config", "get_raw_data", "get_processed_data")
class TestTrainPreprocessing(TestCase):
    def test_train_preprocess(self):
        input_df = self.raw_df
        expected_df = self.processed_df

        train_preprocess = TrainPreprocess(**self.config["train_preprocess"])
        output_df = train_preprocess.preprocess(input_df).sort_values("id").reset_index(drop=True)

        for column in output_df.columns:
            self.assertTrue(expected_df[column].equals(output_df[column]),
                            f"Train preprocessing of column {column} is not correct")
