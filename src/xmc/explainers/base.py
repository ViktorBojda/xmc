from __future__ import annotations
import datetime as dt
import random
import traceback
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
import tensorflow as tf
from alibi.explainers import AnchorTabular

from alibi.explainers.cfproto import CounterfactualProto
from matplotlib import pyplot as plt
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

    @abstractmethod
    def explain_counterfactuals(self) -> None: ...

    @timer
    def run(self) -> None:
        class ExplanationMethod(StrEnum):
            SHAP = "SHAP"
            ANCHORS = "Anchors"
            COUNTERFACTUALS = "Counterfactuals"

        explanation_methods = {
            ExplanationMethod.SHAP: self.explain_shap,
            ExplanationMethod.ANCHORS: self.explain_anchors,
            ExplanationMethod.COUNTERFACTUALS: self.explain_counterfactuals,
        }
        print("Choose which explanation method to run, options are:")
        explanation_method, explanation_name = prompt_options(explanation_methods)
        print(f"Running {explanation_name} explanation...")
        explanation_method()

    def save_shap_explanation(self, explanation: shap.Explanation) -> None:
        path = self.explanations_path / "shap/explanation.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(explanation, filename=path, protocol=5, compress=3)
        print(f"SHAP explanation has been saved to '{path}'.")

    def load_shap_explanation(self) -> shap.Explanation | None:
        path = self.explanations_path / "shap/explanation.joblib"
        if path.exists():
            print(f"Loading SHAP explanation from '{path}'...")
            return joblib.load(path)
        return None

    def plot_shap_explanations(
        self,
        explanation: shap.Explanation,
        y_pred: np.ndarray,
        beeswarm_max_display: int = 30,
    ):
        plt.figure(clear=True)
        for class_idx, class_name in enumerate(self.label_encoder.classes_):
            class_explanation = explanation[:, :, class_idx]
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
                instance_features = self.X_test[instance_idx]
                if self.scaler:
                    instance_features = self.scaler.inverse_transform(
                        instance_features.reshape(1, -1)
                    )
                plt.clf()
                if explanation.base_values.ndim == 2:
                    base_value = explanation.base_values[instance_idx, class_idx]
                else:
                    base_value = explanation.base_values[class_idx]
                shap.plots.decision(
                    base_value,
                    explanation.values[instance_idx, :, class_idx],
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
        base_dir = self.explanations_path / "anchors"
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
            print(f"Anchors creation finished for class {class_name}.\n")
            print("-" * 50)

    def assert_cf_valid(
        self,
        cf_class: int,
        cf_proba: list[float],
        control_class: int,
        control_proba: np.ndarray,
        *,
        rtol: float = 1e-05,
        atol: float = 1e-08,
    ) -> None:
        if cf_class != control_class:
            raise ValueError(
                f"Classes of counterfactual do not match.\nExpected: {cf_class}\nActual: {control_class}"
            )
        if not np.allclose(cf_proba, control_proba, rtol=rtol, atol=atol):
            raise ValueError(
                f"Probabilities of counterfactual and control instance do not match.\n"
                f"Expected: {cf_proba}\nActual: {control_proba}"
            )

    def cf_feature_formatter(
        self,
        predictor: Callable[[np.ndarray], np.ndarray],
        cf_instance: list[float],
        orig_instance: np.ndarray,
        diff_features: np.ndarray,
        cf_class: int,
        cf_proba: list[float],
        rtol: float = 1e-05,
        atol: float = 1e-08,
    ) -> str:
        if self.scaler:
            cf_instance = np.rint(
                self.scaler.inverse_transform(np.array(cf_instance).reshape(1, -1))
            ).astype(int)[0]
            orig_instance = np.rint(
                self.scaler.inverse_transform(orig_instance.reshape(1, -1))
            ).astype(int)[0]
            rtol, atol = 2e-3, 0.2
        result = ""
        for idx in diff_features:
            cf_value = int(cf_instance[idx])
            orig_value = orig_instance[idx]
            if cf_value == orig_value:
                continue
            orig_instance[idx] = cf_value
            result += (
                f"\nFeature: {self.feature_names[idx]} (index {idx})\n"
                f"Value: {cf_value}\nOriginal value: {orig_value}\n"
            )
        orig_instance = orig_instance.reshape(1, -1)
        if self.scaler:
            orig_instance = self.scaler.transform(orig_instance)
        control_proba = predictor(orig_instance)[0]
        control_class = int(np.argmax(control_proba))
        self.assert_cf_valid(
            cf_class, cf_proba, control_class, control_proba, rtol=rtol, atol=atol
        )
        if not result:
            raise ValueError("Something went wrong. No feature changes detected.")
        return result

    def create_counterfactual_explanations(
        self,
        predictor: Callable[[np.ndarray], np.ndarray],
        *,
        explainer_params: dict[str, Any] | None = None,
        samples_per_class: int = 5,
    ):
        explainer_kwargs = {
            "kappa": 0.1,
            "max_iterations": 500,
            "c_steps": 5,
            "eps": (0.001, 0.001),
        }
        if explainer_params:
            explainer_kwargs.update(explainer_params)
        tf.keras.backend.clear_session()
        explainer = CounterfactualProto(
            predictor,
            shape=(1, self.X_train.shape[1]),
            feature_range=(self.X_train.min(axis=0), self.X_train.max(axis=0)),
            use_kdtree=True,
            **explainer_kwargs,
        )
        explainer.fit(self.X_train)
        y_pred = np.argmax(predictor(self.X_test), axis=1)
        base_dir = self.explanations_path / "counterfactuals"
        for class_idx, class_name in enumerate(self.label_encoder.classes_):
            correct_idxs = np.where((self.y_test == class_idx) & (y_pred == class_idx))[
                0
            ]
            if not correct_idxs.size:
                print(
                    f"Failed to create counterfactual, no correct prediction found for class '{class_name}'."
                )
                continue

            random.seed(self.random_state)
            sampled_idxs = random.sample(
                correct_idxs.tolist(), min(samples_per_class, len(correct_idxs))
            )
            if (sample_length := len(sampled_idxs)) != samples_per_class:
                print(
                    f"Failed to find {samples_per_class} correct predictions for class '{class_name}', "
                    f"generating counterfactuals for the {sample_length} samples found."
                )

            for idx in sampled_idxs:
                instance = self.X_test[idx]
                instance_dir = base_dir / f"{class_name}/instance_{idx}"
                instance_dir.mkdir(parents=True, exist_ok=True)
                instance_descriptor = f"instance of class '{class_name}' (index {idx})"
                explanation_json_file = instance_dir / f"explanation.json"
                if explanation_json_file.exists():
                    print(
                        f"Explanation already exists for {instance_descriptor}, skipping."
                    )
                    continue
                try:
                    print(
                        f"Searching for counterfactual for {instance_descriptor}...\n"
                    )
                    # reshape instance to (1, 10_000)
                    explanation = explainer.explain(
                        instance.reshape(1, -1), k=5, verbose=True
                    )
                    explanation_json_file.write_text(explanation.to_json())
                    if cf := getattr(explanation, "cf", None):
                        if len(cf["X"]) != 1:
                            raise ValueError(
                                f"Expected single counterfactual instance, found {len(cf['X'])}"
                            )
                        cf_instance, cf_class, cf_proba = (
                            cf["X"][0],
                            cf["class"],
                            cf["proba"][0],
                        )
                        differences = instance - cf_instance
                        diff_features = np.where(np.abs(differences) > 1e-6)[0]
                        cf_features = self.cf_feature_formatter(
                            predictor,
                            cf_instance,
                            instance,
                            diff_features,
                            cf_class,
                            cf_proba,
                        )
                        result = (
                            f"Counterfactual explanation for {instance_descriptor}:\n"
                            f"Class: {self.label_encoder.classes_[cf_class]}\n"
                            f"Class probabilities: {round_values(cf_proba, 4)}\n\n"
                            f"Original class: {self.label_encoder.classes_[explanation.orig_class]}\n"
                            f"Original class probabilities: {round_values(explanation.orig_proba[0], 4)}\n"
                            f"{cf_features}"
                        )
                        print(result)
                        (instance_dir / f"explanation.txt").write_text(result)
                    else:
                        raise CounterfactualNotFound(
                            f"No counterfactual found for {instance_descriptor}."
                        )
                except Exception as e:
                    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    error_msg = (
                        f"[{timestamp}] Error while generating counterfactual for {instance_descriptor}:\n"
                        f"{str(e)}\n"
                    )
                    print(error_msg)
                    with (base_dir / f"error_logs.txt").open("a") as f:
                        f.write(error_msg)
                        f.write(f"Traceback:\n{traceback.format_exc()}\n")
                        f.write("-" * 50 + "\n")
                print("-" * 50)
            print(f"Counterfactuals creation finished for class '{class_name}'.\n")
            print("-" * 50)


class TreeMalwareExplainer(BaseMalwareExplainer):
    @abstractmethod
    def get_shap_explainer(self) -> shap.TreeExplainer: ...

    @timer
    def explain_shap(self) -> None:
        if not (explanation := self.load_shap_explanation()):
            explanation = self.get_shap_explainer()(self.X_test)
            self.save_shap_explanation(explanation)

        y_pred = self.model.predict(self.X_test)
        self.plot_shap_explanations(explanation, y_pred)

    @timer
    def explain_anchors(self) -> None:
        self.create_anchor_explanations(self.model.predict)
