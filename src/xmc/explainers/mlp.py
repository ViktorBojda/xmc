from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import torch

from xmc.explainers.base import BaseMalwareExplainer
from xmc.utils import timer, stratified_sample, try_import_shap

if TYPE_CHECKING:
    from xmc.classifiers import MalwareClassifierMLP

shap = try_import_shap()


class MalwareExplainerMLP(BaseMalwareExplainer):
    @timer
    def explain_shap(self):
        # Finished MalwareExplainerMLP.run() in 2545.36 secs
        artifacts = self.classifier_class.load_model_artifacts()
        model = artifacts["model"]
        feature_names = artifacts["feature_names"]
        label_encoder = artifacts["label_encoder"]
        X_train, y_train = artifacts["X_train"], artifacts["y_train"]
        X_test, y_test = artifacts["X_test"], artifacts["y_test"]

        X_train_sample = stratified_sample(
            X_train, y_train, size=1000, random_state=self.random_state
        )
        X_train_tensor = torch.from_numpy(X_train_sample)
        X_test_tensor = torch.from_numpy(X_test)
        explainer = shap.DeepExplainer(model, X_train_tensor)
        base_values = explainer.expected_value
        shap_values = explainer.shap_values(X_test_tensor)
        with torch.no_grad():
            outputs = model(X_test_tensor)
            y_pred = torch.argmax(outputs, dim=1).cpu().numpy()

        def explanation_getter(class_idx: int) -> shap.Explanation:
            return shap.Explanation(
                base_values=base_values[class_idx],
                values=shap_values[:, :, class_idx],
                data=X_test_tensor,
                feature_names=feature_names,
            )

        self.plot_shap_explanations(
            explanation_getter, X_test, y_test, y_pred, feature_names, label_encoder
        )

    @timer
    def explain_anchors(self):
        artifacts = self.classifier_class.load_model_artifacts()
        model: MalwareClassifierMLP.MalwareNet = artifacts["model"]
        feature_names = artifacts["feature_names"]
        label_encoder = artifacts["label_encoder"]
        X_train, y_train = artifacts["X_train"], artifacts["y_train"]
        X_test, y_test = artifacts["X_test"], artifacts["y_test"]

        def predictor(X: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32)
                outputs = model(X_tensor)
                return torch.argmax(outputs, dim=1).cpu().numpy()

        self.create_anchors_explanations(
            predictor, X_train, y_train, X_test, y_test, feature_names, label_encoder
        )
