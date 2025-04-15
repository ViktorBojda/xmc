from typing import Any

import numpy as np

from xmc.explainers.base import TreeMalwareExplainer
from xmc.utils import try_import_shap

shap = try_import_shap()


class MalwareExplainerBRF(TreeMalwareExplainer):
    # Finished MalwareExplainerBRF.explain_shap() in 114.74 secs
    def get_shap_explainer(
        self, model: Any, feature_names: list[str]
    ) -> shap.TreeExplainer:
        if shap.__version__ == "0.0.0-not-built":  # local gpu build
            return shap.GPUTreeExplainer(model, feature_names=feature_names)
        return shap.TreeExplainer(model, feature_names=feature_names)

        self.plot_shap_explanations(
            explanation_getter, X_test, y_test, y_pred, feature_names, label_encoder
        )

    @timer
    def explain_anchors(self):
        artifacts = self.classifier_class.load_model_artifacts()
        model: BalancedRandomForestClassifier = artifacts["model"]
        feature_names = artifacts["feature_names"]
        label_encoder = artifacts["label_encoder"]
        X_train, y_train = artifacts["X_train"], artifacts["y_train"]
        X_test, y_test = artifacts["X_test"], artifacts["y_test"]

        self.create_anchors_explanations(
            model.predict,
            X_train,
            y_train,
            X_test,
            y_test,
            feature_names,
            label_encoder,
        )
