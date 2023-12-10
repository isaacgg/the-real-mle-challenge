import argparse
import logging
from typing import Dict

import pandas as pd

from src.data.data_split import DataSplit
from src.data.utils import load_config, save_json
from src.preprocess.preprocess import Preprocess
from src.model.model import RandomForestClassifier
from src.train.evaluation import Evaluation

logger = logging.getLogger(__name__)


class TrainCommand:
    def __init__(self, config: Dict):
        self.preprocess = Preprocess(**config["preprocess"])
        self.data_split = DataSplit(**config["data_split"])
        self.model = RandomForestClassifier(model_path=config["train"]["model_path"],
                                            **config["train"]["params"])
        self.evaluation = Evaluation(classifier=self.model, **config["evaluation"])

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.DataFrame):
        y_pred = self.model.predict(X_test)

        df_report = self.evaluation.get_classification_report(y_test=y_test, y_pred=y_pred)
        accuracy = self.evaluation.get_accuracy(X=X_test, y=y_test)
        roc_auc_score = self.evaluation.get_roc_auc_score(X=X_test, y=y_test)

        self.evaluation.save_confusion_matrix(y_test=y_test, y_pred=y_pred)
        self.evaluation.save_feature_importance()
        self.evaluation.save_metrics(df_report=df_report)

        df_report.to_csv(f"{config['evaluation']['metrics_path']}/reports.csv")
        save_json({"accuracy": accuracy,
                   "roc_auc_score": roc_auc_score},
                  f"{config['evaluation']['metrics_path']}/metrics.json")

    def execute(self):
        data = pd.read_csv(config["base"]["data_path"])

        data = self.preprocess.preprocess(df=data)
        X_train, X_test, y_train, y_test = self.data_split.split(data)

        self.model.fit(X_train, y_train)
        self.evaluate(X_test, y_test)


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--config', dest='config', required=True)
    args = argument_parser.parse_args()

    config = load_config(args.config)

    TrainCommand(config).execute()
