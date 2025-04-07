from imblearn.ensemble import BalancedRandomForestClassifier

from xmc.explainers.base import BaseMalwareExplainer
from xmc.utils import timer, try_import_shap

shap = try_import_shap()


class MalwareExplainerBRF(BaseMalwareExplainer):
    @timer
    def explain_shap(self):
        # Finished MalwareExplainerBRF.explain_shap() in 114.74 secs
        artifacts = self.classifier_class.load_model_artifacts()
        model = artifacts["model"]
        feature_names = artifacts["feature_names"]
        label_encoder = artifacts["label_encoder"]
        X_test, y_test = artifacts["X_test"], artifacts["y_test"]

        if shap.__version__ == "0.0.0-not-built":  # local gpu build
            explainer = shap.GPUTreeExplainer(model, feature_names=feature_names)
        else:
            explainer = shap.TreeExplainer(model, feature_names=feature_names)
        explanation = explainer(X_test)
        y_pred = model.predict(X_test)

        def explanation_getter(class_idx: int) -> shap.Explanation:
            return explanation[:, :, class_idx]

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
