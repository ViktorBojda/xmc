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
