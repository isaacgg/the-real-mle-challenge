import os.path
from typing import List, Dict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

from src.model.abc.classifier import Classifier


class Evaluation:
    feature_importance_file = "feature_importance.png"
    confusion_matrix_file = "confusion_matrix.png"
    metrics_file = "metrics.png"

    def __init__(self, classifier: Classifier, metrics_path: str, metrics: List[str], report_maps: Dict[str, str]):
        self.classifier = classifier
        self.figures_path = f"{metrics_path}/figures/"
        self.metrics = metrics
        self.report_maps = report_maps

    def get_accuracy(self, X: pd.DataFrame, y: pd.DataFrame) -> float:
        y_pred = self.classifier.predict(X)
        return accuracy_score(y, y_pred)

    def get_roc_auc_score(self, X: pd.DataFrame, y: pd.DataFrame) -> float:
        y_proba = self.classifier.predict_proba(X)
        return roc_auc_score(y, y_proba, multi_class='ovr')

    def save_feature_importance(self):
        features_importance = self.classifier.get_feature_importance()
        features, importance = list(features_importance.keys()), list(features_importance.values())

        fig, ax = plt.subplots(figsize=(12, 7))
        plt.barh(range(len(importance)), list(importance))
        plt.yticks(range(len(importance)), features, fontsize=12)
        ax.invert_yaxis()
        ax.set_xlabel("Feature importance", fontsize=12)
        plt.suptitle(self.classifier.name, fontsize=14)

        plt.savefig(os.path.join(self.figures_path, self.feature_importance_file))

    def save_confusion_matrix(self, y_test: pd.DataFrame, y_pred: pd.DataFrame):
        classes = [0, 1, 2, 3]
        labels = ['low', 'mid', 'high', 'lux']

        c = confusion_matrix(y_test, y_pred)
        c = c / c.sum(axis=1).reshape(len(classes), 1)

        # Plot
        # fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(c, annot=True, cmap='BuGn', square=True, fmt='.2f', annot_kws={'size': 10}, cbar=False)
        plt.xlabel('Predicted', fontsize=16)
        plt.ylabel('Real', fontsize=16)
        plt.xticks(ticks=np.arange(.5, len(classes)), labels=labels, rotation=0, fontsize=12)
        plt.yticks(ticks=np.arange(.5, len(classes)), labels=labels, rotation=0, fontsize=12)
        plt.title(self.classifier.name, fontsize=18)

        plt.savefig(os.path.join(self.figures_path, self.confusion_matrix_file))

    def get_classification_report(self, y_test: pd.DataFrame, y_pred: pd.DataFrame) -> pd.DataFrame:
        report = classification_report(y_test, y_pred, output_dict=True)
        df_report = pd.DataFrame.from_dict(report).T[:-3]
        df_report.index = [self.report_maps[i] for i in df_report.index]
        return df_report

    def save_metrics(self, df_report: pd.DataFrame):
        fig, axes = plt.subplots(1, len(self.metrics), figsize=(16, 7))

        for i, ax in enumerate(axes):
            ax.barh(df_report.index, df_report[self.metrics[i]], alpha=0.9)
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.set_xlabel(self.metrics[i], fontsize=12)
            ax.invert_yaxis()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        plt.suptitle(self.classifier.name, fontsize=14)

        plt.savefig(os.path.join(self.figures_path, self.metrics_file))
