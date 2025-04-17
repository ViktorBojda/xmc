from imblearn.ensemble import BalancedRandomForestClassifier

from xmc.explainers.base import TreeMalwareExplainer
from xmc.utils import try_import_shap, timer

shap = try_import_shap()


class MalwareExplainerBRF(TreeMalwareExplainer):
    model: BalancedRandomForestClassifier

    def get_shap_explainer(self) -> shap.GPUTreeExplainer | shap.TreeExplainer:
        if shap.__version__ == "0.0.0-not-built":  # local gpu build
            return shap.GPUTreeExplainer(self.model, feature_names=self.feature_names)
        return shap.TreeExplainer(self.model, feature_names=self.feature_names)

    @timer
    def explain_counterfactuals(self) -> None:
        self.create_counterfactual_explanations(self.model.predict_proba)
