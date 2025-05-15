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
        # Finished MalwareExplainerXGB.explain_counterfactuals() in 6565.18 secs
        self.create_counterfactual_explanations(
            self.model.predict_proba,
            explainer_params={"kappa": 0.01, "beta": 1.0, "c_init": 5.0},
        )
