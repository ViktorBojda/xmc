from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import torch

from xmc.explainers.base import BaseMalwareExplainer
from xmc.utils import timer, stratified_sample, try_import_shap

if TYPE_CHECKING:
    from xmc.classifiers import MalwareClassifierMLP

shap = try_import_shap()


class MalwareExplainerMLP(BaseMalwareExplainer):
    model: MalwareClassifierMLP.MalwareNet

    def __init__(self, *args, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(*args, **kwargs)

    @timer
    def explain_shap(self):
        # Finished MalwareExplainerMLP.explain_shap() in 51.87 secs
        X_test_tensor = torch.from_numpy(self.X_test).float().to(self.device)
        if not (explanation := self.load_shap_explanation()):
            X_train_sample = stratified_sample(
                self.X_train, self.y_train, size=1000, random_state=self.random_state
            )
            X_train_tensor = torch.from_numpy(X_train_sample).float().to(self.device)
            explainer = shap.DeepExplainer(self.model, X_train_tensor)
            base_values = explainer.expected_value
            # can take more than 1 hour
            shap_values = explainer.shap_values(X_test_tensor, check_additivity=False)
            explanation = shap.Explanation(
                base_values=base_values,
                values=shap_values,
                data=X_test_tensor,
                feature_names=self.feature_names,
            )
            self.save_shap_explanation(explanation)

        with torch.no_grad():
            outputs = self.model(X_test_tensor)
            y_pred = torch.argmax(outputs, dim=1).cpu().numpy()

        explanation.data = self.scaler.inverse_transform(self.X_test)
        self.plot_shap_explanations(explanation, y_pred)

    @timer
    def explain_anchors(self):
        vocabulary = self.vectorizer.vocabulary_
        n_features = len(self.feature_names)

        def predictor(X: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
                outputs = self.model(X_tensor)
                return torch.argmax(outputs, dim=1).cpu().numpy()

        def anchor_formatter(anchor: list[str]) -> str:
            new_anchor = []
            for rule in anchor:
                match = re.search(r"^(.*?)\s*([<>=]=?)\s*([\d.]+)$", rule)
                if not match:
                    raise ValueError(
                        f"Anchor rule is not in expected format, expected: 'FEATURE OPERATOR VALUE'\nactual: '{rule}'."
                    )
                feature = match.group(1)
                operator = match.group(2)
                value = float(match.group(3))
                feature_idx = 0, vocabulary[feature]
                dummy_row = np.zeros((1, n_features))
                dummy_row[feature_idx] = value
                inverse_value = int(
                    self.scaler.inverse_transform(dummy_row)[feature_idx]
                )
                new_anchor.append(f"{feature} {operator} {inverse_value}")
            return self.join_anchor_rules(new_anchor)

        self.create_anchor_explanations(predictor, anchor_formatter)

    @timer
    def explain_counterfactuals(self) -> None:
        # Finished MalwareExplainerMLP.explain_counterfactuals() in 2180.01 secs
        def predictor(X: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
                outputs = self.model(X_tensor)
                return outputs.cpu().numpy()

        self.create_counterfactual_explanations(
            predictor, explainer_params={"kappa": 1.0}
        )
