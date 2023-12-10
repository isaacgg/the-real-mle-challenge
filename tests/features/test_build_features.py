from typing import Optional, Union
from unittest import TestCase

import numpy as np
import pandas as pd
from parameterized import parameterized

from src.preprocess.preprocess import Preprocess


class TestPreprocessing(TestCase):
    @parameterized([("1 bath", 1), ("1.5 bath", 1.5), (np.NaN, np.NaN), (None, np.NaN)])
    def test_num_bathroom_from_text(self, number: Optional[Union[str, float]], expected: float):
        output = Preprocess()._num_bathroom_from_text(number)
        self.assertEquals(output, expected)

    def test_price_to_int(self):
        df = pd.DataFrame([("$150.00", 150),
                           ("$75.00", 75),
                           ("$60.00", 60),
                           ("$275.00", 275),
                           ("$68.00", 68)], columns=["price_str", "expected"])
        output = Preprocess()._price_to_int(df["price_str"])
        self.assertEquals(output, df["expected"])

