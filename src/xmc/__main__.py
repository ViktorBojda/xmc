from enum import StrEnum

from xmc.classifiers import (
    MalwareClassifierBRF,
    MalwareClassifierMLP,
    MalwareClassifierXGB,
)
from xmc.utils import prompt_options


class ClassifierType(StrEnum):
    BRF = "Balanced Random Forest"
    XGB = "XGBoost"
    MLP = "Multilayer Perceptron"


classifiers = {
    ClassifierType.BRF: MalwareClassifierBRF,
    ClassifierType.XGB: MalwareClassifierXGB,
    ClassifierType.MLP: MalwareClassifierMLP,
}

if __name__ == "__main__":
    print("Choose which classifier to run, options are:")
    clf, clf_name = prompt_options(classifiers)
    print(f"Running {clf_name} classifier...")
    clf.run()
