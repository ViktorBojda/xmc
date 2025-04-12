from typing import Any

import numpy as np
from imblearn.ensemble import BalancedRandomForestClassifier
from scipy.sparse import csc_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

from xmc.classifiers.base import BaseMalwareClassifier
from xmc.explainers.brf import MalwareExplainerBRF
from xmc.utils import timer

# Cross-Validation f1_macro scores: [0.7464, 0.7538, 0.7836, 0.7654, 0.7441, 0.7382, 0.7591, 0.7694, 0.7599, 0.7337]
# Cross-Validation f1_macro mean:   0.7554
# Cross-Validation f1_macro std:    0.0153
# --------------------------------------------------
# Finished MalwareClassifierBRF.cross_validate() in 713.47 secs
# Classification Report:
#                precision    recall  f1-score   support
#
#       adware       0.89      0.62      0.73       279
#     backdoor       0.91      0.76      0.82       366
#   downloader       0.76      0.78      0.77       225
#      dropper       0.57      0.73      0.64       169
#      spyware       0.50      0.58      0.54       160
#       trojan       0.89      0.98      0.93      2913
#        virus       0.96      0.88      0.92      1097
#        worms       0.84      0.58      0.68       359
#
#     accuracy                           0.87      5568
#    macro avg       0.79      0.74      0.76      5568
# weighted avg       0.88      0.87      0.87      5568
#
# --------------------------------------------------
# Finished MalwareClassifierBRF.train_and_evaluate() in 230.98 secs


class MalwareClassifierBRF(BaseMalwareClassifier):
    model_name = "brf"
    explainer_class = MalwareExplainerBRF

    def __init__(
        self,
        max_features: int = 1_000,
        ngram_range: tuple[int, int] = (1, 2),
        n_estimators: int = 250,
        max_depth: int = 31,
        replacement: bool = True,
        bootstrap: bool = False,
        sampling_strategy: str = "not majority",
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        rf_max_features: str | None = None,
        random_state: int = 69,
        verbose: int = 2,
        n_jobs: int = -1,
    ):
        self.random_state = random_state
        self.vectorizer = CountVectorizer(
            tokenizer=self.comma_tokenizer,
            token_pattern=None,
            max_features=max_features,
            ngram_range=ngram_range,
        )
        self.label_encoder = LabelEncoder()
        self.classifier = BalancedRandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            replacement=replacement,
            bootstrap=bootstrap,
            sampling_strategy=sampling_strategy,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=rf_max_features,
            random_state=self.random_state,
            verbose=verbose,
            n_jobs=n_jobs,
        )

    @timer
    def cross_validate(
        self,
        X: csc_matrix,
        y: np.ndarray,
        *,
        cv_splits: int = 10,
        scoring: str = "f1_macro",
    ) -> None:
        """
        Performs Stratified K-Fold cross-validation and prints the scores.
        """
        kfold = StratifiedKFold(
            n_splits=cv_splits, shuffle=True, random_state=self.random_state
        )
        cv_scores = cross_val_score(
            self.classifier, X, y, cv=kfold, scoring=scoring, verbose=1, n_jobs=-1
        )
        self.display_cv_results(scoring, cv_scores)

    @timer
    def train_and_evaluate(
        self, X: np.ndarray, y: np.ndarray, *, test_size: float = 0.2
    ) -> None:
        """
        Splits data into train/test, fits the classifier on training set,
        predicts on the test set, decodes the labels, and prints a classification report.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        self.classifier.fit(X_train, y_train)
        y_pred_encoded = self.classifier.predict(X_test)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        y_true = self.label_encoder.inverse_transform(y_test)
        print("Classification Report:\n", classification_report(y_true, y_pred))
        self.plot_confusion_matrix(y_true, y_pred)
        print("-" * 50)
        self.save_model_artifacts(X_train, X_test, y_train, y_test)

    def get_model_artifacts(self) -> dict[str, Any]:
        artifacts = super().get_model_artifacts()
        self.classifier.set_params(verbose=0)
        artifacts["model"] = self.classifier
        return artifacts
