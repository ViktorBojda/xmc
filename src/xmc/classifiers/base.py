import inspect
from abc import ABC, abstractmethod
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.sparse import csc_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from xmc.settings import MODELS_DIR_PATH
from xmc.explainers.base import BaseMalwareExplainer
from xmc.utils import prompt_overwrite, timer, prompt_options, save_plot, load_dataset


class BaseMalwareClassifier(ABC):
    DATASET_NAME = "malware_dataset.csv"

    def __init__(
        self,
        max_features: int,
        ngram_range: tuple[int, int],
        use_scaler: bool = False,
        random_state: int = 69,
    ) -> None:
        self.vectorizer = CountVectorizer(
            tokenizer=self.comma_tokenizer,
            token_pattern=None,
            max_features=max_features,
            ngram_range=ngram_range,
        )
        self.label_encoder = LabelEncoder()
        self.scaler = MinMaxScaler() if use_scaler else None
        self.random_state = random_state
        self.reset_score_metrics()
        self.init_params = self.get_model_architecture()

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @classmethod
    def model_path(cls) -> Path:
        return MODELS_DIR_PATH / f"{cls.model_name}.joblib"

    @property
    @abstractmethod
    def explainer_class(self) -> type[BaseMalwareExplainer]: ...

    def get_model_architecture(self) -> str:
        return ", ".join(
            [str(x) for x in (inspect.signature(self.__init__).parameters.values())]
        )

    def plot_confusion_matrix(self, y_true: pd.Series, y_pred: pd.Series) -> None:
        classes = self.label_encoder.classes_
        cm = confusion_matrix(y_true, y_pred, labels=classes, normalize="true")
        _, ax = plt.subplots(figsize=(5, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot(ax=ax, values_format=".2f")
        plt.xticks(rotation=45)
        save_plot(
            title=f"{self.model_name.upper()} Confusion Matrix",
            save_as=f"{self.model_name.lower()}/confusion_matrix",
        )

    @staticmethod
    def comma_tokenizer(text: str) -> list[str]:
        return text.split(",")

    def load_and_transform_data(self) -> tuple[csc_matrix, np.ndarray]:
        df = load_dataset(self.DATASET_NAME)
        X = csc_matrix(self.vectorizer.fit_transform(df["api"]))
        y = self.label_encoder.fit_transform(df["class"])
        return X, y

    def log_write(self, text: str) -> None:
        log_text = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n{text}"
        print(log_text)
        with (MODELS_DIR_PATH / f"{self.model_name}_log.txt").open("a") as f:
            f.write(log_text)

    def save_init_params(self):
        ", ".join(
            [str(x) for x in (inspect.signature(self.__init__).parameters.values())]
        )

    def reset_score_metrics(self) -> None:
        self.score_metrics = {
            "accuracy": [],
            "precision_weighted": [],
            "precision_macro": [],
            "recall_weighted": [],
            "recall_macro": [],
            "f1_weighted": [],
            "f1_macro": [],
        }

    def calc_score_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        self.score_metrics["accuracy"].append(accuracy_score(y_true, y_pred))
        self.score_metrics["precision_macro"].append(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        )
        self.score_metrics["precision_weighted"].append(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        )
        self.score_metrics["recall_macro"].append(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        )
        self.score_metrics["recall_weighted"].append(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        )
        self.score_metrics["f1_macro"].append(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        )
        self.score_metrics["f1_weighted"].append(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        )

    def log_score_metrics(
        self, params: Any | None = None, *, weighted: bool = False
    ) -> None:
        text = (
            f"Architecture: {params if params else self.init_params}\n"
            f"Cross-Validation Results Summary:\n"
            f"{'Metric':<25}{'Mean':<10}SD\n"
        )
        for metric, values in self.score_metrics.items():
            if not weighted and "weighted" in metric:
                continue
            text += (
                f"{metric:<25}{np.mean(values):<10.4f}{np.std(values, ddof=1):.4f}\n"
            )
        text += "LaTeX table row:\n"
        for metric, values in self.score_metrics.items():
            if not weighted and "weighted" in metric:
                continue
            text += f"{np.mean(values):.3f}\pm{np.std(values, ddof=1):.3f} & "
        text = text[:-2] + "\\\\\n"
        text += "–" * 50 + "\n"
        self.log_write(text)

    @abstractmethod
    def cross_validate(self, X, y, *, cv_splits): ...

    @abstractmethod
    def train_and_evaluate(self, X, y, *, test_size): ...

    def get_model_artifacts(self) -> dict[str, Any]:
        feature_names = list(self.vectorizer.get_feature_names_out())
        artifacts = {
            "vectorizer": self.vectorizer,
            "label_encoder": self.label_encoder,
            "feature_names": feature_names,
        }
        if self.scaler:
            artifacts["scaler"] = self.scaler
        return artifacts

    def _save_model_artifacts(self, artifacts: dict[str, Any], path: Path) -> None:
        joblib.dump(artifacts, filename=path, protocol=5, compress=3)

    def save_model_artifacts(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        model_path = self.model_path()
        if not prompt_overwrite(model_path):
            return
        artifacts = self.get_model_artifacts()
        artifacts.update(
            {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}
        )
        self._save_model_artifacts(artifacts, model_path)
        print(f"Model artifacts have been saved to: {model_path}")

    @classmethod
    def _load_model_artifacts(cls) -> dict[str, Any]:
        return joblib.load(cls.model_path())

    @classmethod
    def load_model_artifacts(cls) -> dict[str, Any]:
        print(f"Loading artifacts for model '{cls.model_name}'...")
        try:
            return cls._load_model_artifacts()
        except FileNotFoundError:
            print(
                f"ERROR: No model artifacts found for '{cls.model_name}' model, "
                f"make sure you run 'Train & Evaluate' first."
            )
            exit(1)

    @classmethod
    def run_explainer(cls) -> None:
        explainer_class = cls.explainer_class
        explainer_class(classifier_class=cls)

    @classmethod
    @timer
    def run(cls) -> None:
        class RunMethod(StrEnum):
            CV = "Cross Validation"
            TRAIN = "Train & Evaluate"
            EXPLAIN = "Explain"
            TUNE = "Tune"  # TODO: temporary

        print("Choose which classifier method(s) to run, options are:")
        method_names, display_names = prompt_options(
            RunMethod._value2member_map_, multi_select=True
        )
        if RunMethod.TUNE in method_names:
            print(f"Running {RunMethod.TUNE} method...")
            instance = cls()
            instance._tune_hyperparameters()
            return
        if RunMethod.CV in method_names or RunMethod.TRAIN in method_names:
            instance = cls()
            X, y = instance.load_and_transform_data()
            if RunMethod.CV in method_names:
                print(f"Running {RunMethod.CV} method...")
                instance.cross_validate(X, y, cv_splits=2)
            if RunMethod.TRAIN in method_names:
                print(f"Running {RunMethod.TRAIN} method...")
                X = X.toarray() if hasattr(X, "toarray") else X
                instance.train_and_evaluate(X, y, test_size=0.2)
        if RunMethod.EXPLAIN in method_names:
            print(f"Running {RunMethod.EXPLAIN} method...")
            cls.run_explainer()
