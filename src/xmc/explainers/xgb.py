from xgboost import XGBClassifier

from xmc.explainers.base import TreeMalwareExplainer
from xmc.utils import try_import_shap

shap = try_import_shap()


class MalwareExplainerXGB(TreeMalwareExplainer):
    model: XGBClassifier

    # Finished MalwareExplainerXGB.explain_shap() in 58.88 secs
    def get_shap_explainer(self) -> shap.TreeExplainer:
        return shap.TreeExplainer(self.model, feature_names=self.feature_names)

