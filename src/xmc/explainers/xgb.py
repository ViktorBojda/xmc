from typing import Any

from xmc.explainers.base import TreeMalwareExplainer
from xmc.utils import try_import_shap

shap = try_import_shap()


class MalwareExplainerXGB(TreeMalwareExplainer):
    # Finished MalwareExplainerXGB.explain_shap() in 58.88 secs
    def get_shap_explainer(
        self, model: Any, feature_names: list[str]
    ) -> shap.TreeExplainer:
        return shap.TreeExplainer(model, feature_names=feature_names)

    @timer
    def explain_anchors(self) -> None:
        artifacts = self.classifier_class.load_model_artifacts()
        model: XGBClassifier = artifacts["model"]
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
