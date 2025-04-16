from __future__ import annotations
import datetime as dt
import random
import traceback
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np
from alibi.explainers import AnchorTabular
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils import compute_class_weight

from xmc.classifiers.utils import save_plot
from xmc.exceptions import CounterfactualNotFound, AnchorNotFound
from xmc.settings import EXPLANATIONS_DIR_PATH
from xmc.utils import prompt_options, try_import_shap, round_values, timer

if TYPE_CHECKING:
    from xmc.classifiers.base import BaseMalwareClassifier

shap = try_import_shap()


class BaseMalwareExplainer(ABC):
    def __init__(
        self, classifier_class: type[BaseMalwareClassifier], random_state: int = 69
    ) -> None:
        self.classifier_class = classifier_class
        self.random_state = random_state
        artifacts = self.classifier_class.load_model_artifacts()
        self.model = artifacts["model"]
        self.vectorizer: CountVectorizer = artifacts["vectorizer"]
        self.feature_names: list[str] = artifacts["feature_names"]
        self.label_encoder: LabelEncoder = artifacts["label_encoder"]
        self.scaler: MinMaxScaler | None = artifacts.get("scaler")
        self.X_train: np.ndarray = artifacts["X_train"]
        self.y_train: np.ndarray = artifacts["y_train"]
        self.X_test: np.ndarray = artifacts["X_test"]
        self.y_test: np.ndarray = artifacts["y_test"]
        self.run()

    @property
    def explanations_path(self) -> Path:
        return EXPLANATIONS_DIR_PATH / f"{self.classifier_class.model_name}"

    @abstractmethod
    def explain_shap(self) -> None: ...

    @abstractmethod
    def explain_anchors(self) -> None: ...

    def run(self) -> None:
        class ExplanationMethod(StrEnum):
            SHAP = "SHAP"
            ANCHORS = "Anchors"

        explanation_methods = {
            ExplanationMethod.SHAP: self.explain_shap,
            ExplanationMethod.ANCHORS: self.explain_anchors,
        }
        print("Choose which explanation method to run, options are:")
        explanation_method, explanation_name = prompt_options(explanation_methods)
        print(f"Running {explanation_name} explanation...")
        explanation_method()

    def load_explanation(self, path: Path) -> Explanation | None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return Explanation.from_json(path.open().read())
        return None

    def plot_shap_explanations(
        self,
        explanation_getter: Callable[[int], shap.Explanation],
        y_pred: np.ndarray,
        beeswarm_max_display: int = 30,
    ):
        plt.figure(clear=True)
        for class_idx, class_name in enumerate(self.label_encoder.classes_):
            class_explanation = explanation_getter(class_idx)
            plt.clf()
            shap.plots.beeswarm(
                class_explanation,
                max_display=beeswarm_max_display,
                group_remaining_features=False,
                show=False,
            )
            save_plot(
                f"SHAP Beeswarm Plot for Class: {class_name} (Mean Avg)",
                f"{self.classifier_class.model_name}/beeswarm/mean_avg/{class_name}",
            )
            max_positive_shap = np.max(
                np.where(class_explanation.values > 0, class_explanation.values, 0),
                axis=0,
            )
            order = np.argsort(-max_positive_shap)
            plt.clf()
            shap.plots.beeswarm(
                class_explanation,
                max_display=beeswarm_max_display,
                order=order,
                group_remaining_features=False,
                show=False,
            )
            save_plot(
                f"SHAP Beeswarm Plot for Class: {class_name} (Max Positive)",
                f"{self.classifier_class.model_name}/beeswarm/max_positive/{class_name}",
            )

            def create_decision_plot(instance_idxs: np.ndarray, identifier: str):
                if instance_idxs.size == 0:
                    return print(
                        f"Failed to create decision plot, no correct prediction found for class {class_name}."
                    )
                instance_idx = instance_idxs[0]
                instance_explanation = class_explanation[instance_idx]
                instance_features = self.X_test[instance_idx]
                plt.clf()
                shap.plots.decision(
                    instance_explanation.base_values,
                    instance_explanation.values,
                    instance_features,
                    self.feature_names,
                    show=False,
                )
                save_plot(
                    f"SHAP Decision Plot for Class: {class_name} ({identifier.capitalize()})",
                    f"{self.classifier_class.model_name}/decision/{identifier.lower()}/{class_name}",
                )

            correct_idxs = np.where((self.y_test == class_idx) & (y_pred == class_idx))
            create_decision_plot(correct_idxs[0], "correct")
            incorrect_idxs = np.where(
                (self.y_test == class_idx) & (y_pred != class_idx)
            )
            create_decision_plot(incorrect_idxs[0], "misclassified")

            print(f"SHAP plots created for class {class_name}.")
        plt.close("all")

    def join_anchor_rules(self, anchor: list[str]) -> str:
        return "\nAND ".join(anchor)

    def create_anchor_explanations(
        self,
        predictor: Callable[[np.ndarray], np.ndarray],
        anchor_formatter: Callable[[list[str]], str] | None = None,
        samples_per_class: int = 5,
    ):
        if not anchor_formatter:
            anchor_formatter = self.join_anchor_rules
        y_pred = predictor(self.X_test)
        class_weights = compute_class_weight(
            class_weight="balanced", classes=np.unique(self.y_train), y=self.y_train
        )
        thresholds = {"strict": 0.9, "general": 0.8}
        base_dir = EXPLANATIONS_DIR_PATH / f"{self.classifier_class.model_name}/anchors"
        total_count = self.y_test.shape[0]
        for class_idx, class_name in enumerate(self.label_encoder.classes_):
            total_class_count = np.sum(self.y_test == class_idx)
            correct_idxs = np.where((self.y_test == class_idx) & (y_pred == class_idx))[
                0
            ]
            if not correct_idxs.size:
                print(
                    f"Failed to create anchors, no correct prediction found for class '{class_name}'."
                )
                continue

            random.seed(self.random_state)
            sampled_idxs = random.sample(
                correct_idxs.tolist(), k=min(samples_per_class, len(correct_idxs))
            )
            if (sample_length := len(sampled_idxs)) != samples_per_class:
                print(
                    f"Failed to find {samples_per_class} correct predictions for class '{class_name}', "
                    f"generating anchors for the {sample_length} samples found."
                )

            for idx in sampled_idxs:
                instance = self.X_test[idx]
                instance_dir = base_dir / f"{class_name}/instance_{idx}"
                instance_dir.mkdir(parents=True, exist_ok=True)

                for mode, threshold in thresholds.items():
                    instance_descriptor = (
                        f"instance of class '{class_name}', index {idx}, mode '{mode}'"
                    )
                    explanation_json_file = instance_dir / f"explanation_{mode}.json"
                    if explanation_json_file.exists():
                        print(
                            f"Explanation already exists for {instance_descriptor}, skipping."
                        )
                        continue
                    try:
                        # reinit the explainer to free up memory, as it doesn't release it properly otherwise
                        explainer = AnchorTabular(
                            predictor=predictor,
                            feature_names=self.feature_names,
                            seed=self.random_state,
                        )
                        explainer.fit(self.X_train)
                        explanation = explainer.explain(
                            instance,
                            threshold=threshold,
                            beam_size=1,
                            max_anchor_size=15,
                            verbose_every=5,
                            verbose=True,
                        )
                        explanation_json_file.write_text(explanation.to_json())
                        if anchor := getattr(explanation, "anchor", None):
                            coverage: float = explanation.coverage
                            result = (
                                f"Anchors explanation for {instance_descriptor}:\n"
                                f"Precision: {round(explanation.precision, 4)}\n"
                                f"Coverage: {coverage}\n"
                                f"Weighted coverage: {round(class_weights[class_idx] * coverage, 4)}\n"
                                f"Class coverage: {round(total_count * coverage / total_class_count, 4)}\n"
                                f"Anchor:\nIF {anchor_formatter(anchor)}\nTHEN PREDICT {class_name}\n"
                            )
                            print(result)
                            (instance_dir / f"explanation_{mode}.txt").write_text(
                                result
                            )
                        else:
                            raise AnchorNotFound(
                                f"No anchor found for {instance_descriptor}."
                            )
                    except Exception as e:
                        timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        error_msg = (
                            f"[{timestamp}] Error while generating anchor for {instance_descriptor}:\n"
                            f"{str(e)}\n"
                        )
                        print(error_msg)
                        with (base_dir / f"error_logs.txt").open("a") as f:
                            f.write(error_msg)
                            f.write(f"Traceback:\n{traceback.format_exc()}\n")
                            f.write("-" * 50 + "\n")
                    print("-" * 50)
            print(f"Anchors created for class {class_name}.\n")
            print(f"Anchors creation finished for class {class_name}.\n")
            print("-" * 50)

            print("-" * 50)


class TreeMalwareExplainer(BaseMalwareExplainer):
    @abstractmethod
    def get_shap_explainer(self) -> shap.TreeExplainer: ...

    @timer
    def explain_shap(self):
        explainer = self.get_shap_explainer()
        explanation = explainer(self.X_test)
        y_pred = self.model.predict(self.X_test)

        def explanation_getter(class_idx: int) -> shap.Explanation:
            return explanation[:, :, class_idx]

        self.plot_shap_explanations(explanation_getter, y_pred)

    @timer
    def explain_anchors(self) -> None:
        self.create_anchor_explanations(self.model.predict)
