import numpy as np
from mne.decoding import CSP, Vectorizer
from pyriemann.classification import MDM
from pyriemann.estimation import ERPCovariances
from pyriemann.spatialfilters import Xdawn
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from somepytools.typing import Any, Dict


class PyRimanianClassifier:
    def __init__(self) -> None:
        self.classifier = {
            "LR": (
                make_pipeline(Vectorizer(), LogisticRegression()),
                {"logisticregression__C": np.exp(np.linspace(-4, 4, 5))},
            ),
            "LDA": (
                make_pipeline(Vectorizer(), LDA(shrinkage="auto", solver="eigen")),
                {},
            ),
            "SVM": (
                make_pipeline(Vectorizer(), SVC()),
                {
                    "svc__C": np.exp(np.linspace(-4, 4, 5)),
                    "svc__kernel": ("linear", "rbf"),
                },
            ),
            "CSP LDA": (
                make_pipeline(CSP(), LDA(shrinkage="auto", solver="eigen")),
                {"csp__n_components": (6, 9, 13), "csp__cov_est": ("concat", "epoch")},
            ),
            "Xdawn LDA": (
                make_pipeline(
                    Xdawn(2, classes=[1]),
                    Vectorizer(),
                    LDA(shrinkage="auto", solver="eigen"),
                ),
                {},
            ),
            "ERPCov TS LR": (
                make_pipeline(
                    ERPCovariances(estimator="oas"), TangentSpace(), LogisticRegression()
                ),
                {"erpcovariances__estimator": ("lwf", "oas")},
            ),
            "ERPCov MDM": (
                make_pipeline(ERPCovariances(), MDM()),
                {"erpcovariances__estimator": ("lwf", "oas")},
            ),
        }

    def select_classifier(self, name: str) -> Any:
        return self.classifier[name][0]

    def set_of_parameters(self, name: str) -> Dict[str, Any]:
        return self.classifier[name][1]
