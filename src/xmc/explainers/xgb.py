from xgboost import XGBClassifier

from xmc.explainers.base import TreeMalwareExplainer
from xmc.utils import try_import_shap, timer

shap = try_import_shap()


class MalwareExplainerXGB(TreeMalwareExplainer):
    model: XGBClassifier

    def get_shap_explainer(self) -> shap.TreeExplainer:
        return shap.TreeExplainer(self.model, feature_names=self.feature_names)

    @timer
    def explain_counterfactuals(self, **kwargs) -> None:
        self.create_counterfactual_explanations(self.model.predict_proba)
