from importlib import reload

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from somepytools.typing import Any

import transforms
from eeg_models.datasets.braininvaders import BrainInvadersDataset
from eeg_models.pyrimanianclassifier import PyRimanianClassifier
from transforms import ButterFilter, ChannellwiseScaler, Decimator, MarkersTransformer


class TrainPyRimanianClassifier:
    names = (
        "LR",
        "LDA",
        "SVM",
        "CSP LDA",
        "Xdawn LDA",
        "ERPCov TS LR",
        "ERPCov MDM",
    )
    scores = (
        "balanced_accuracy",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    )

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset = None

    def classifier(self, name: str) -> Any:
        clf = PyRimanianClassifier.select_classifier(name)
        return clf

    def set_dataset(
        self,
        decimation_factor,
        epoch_duration,
        sampling_rate,
        filter,
    ):
        order, highpass, lowpass = filter
        print(
            ">> Decimation factor : ",
            decimation_factor,
            "- epoch_duration : ",
            epoch_duration,
            "- order : ",
            order,
            "-highpass : ",
            highpass,
            " - lowpass : ",
            lowpass,
        )
        final_rate = sampling_rate // decimation_factor
        labels_mapping = {33285.0: 1, 33286.0: 0}
        reload(transforms)
        eeg_pipe = make_pipeline(
            Decimator(decimation_factor),
            ButterFilter(sampling_rate // decimation_factor, order, highpass, lowpass),
            ChannellwiseScaler(StandardScaler()),
        )
        markers_pipe = MarkersTransformer(labels_mapping, decimation_factor)
        epoch_count = int(epoch_duration * final_rate)

        raw_dataset = BrainInvadersDataset()
        for i in range(1, 1 + len(raw_dataset)):
            eeg_pipe.fit(raw_dataset[i]["eegs"])

        dataset = []
        for i in range(1, 1 + len(raw_dataset)):
            epochs = []
            labels = []
            filtered = eeg_pipe.transform(raw_dataset[i]["eegs"])  # seconds
            markups = markers_pipe.transform(raw_dataset[i]["markers"])
            for signal, markup in zip(filtered, markups):
                epochs.extend(
                    [signal[:, start : (start + epoch_count)] for start in markup[:, 0]]
                )
                labels.extend(markup[:, 1])
            dataset.append((np.array(epochs), np.array(labels)))

        (eegs, markers) = dataset[0]
        for i in range(1, len(dataset)):
            eegs = np.append(eegs, dataset[i][0], axis=0)
            markers = np.append(markers, dataset[i][1], axis=0)

        self.dataset = (eegs, markers)

    def crossvalidate_record(self):
        record = self.dataset
        scores = self.scores
        df = pd.DataFrame()
        for name in self.names:
            (clf, params) = self.classifier(name=name)
            cv = GridSearchCV(
                clf,
                params,
                scoring=scores,
                n_jobs=-1,
                refit=False,
                cv=4,
            )
            cv.fit(record[0], record[1])
            headers = [
                name
                for name in cv.cv_results_.keys()
                if name.startswith("param_")
                or name.startswith("mean_test_")
                or name.startswith("std_test_")
            ]
            results = pd.DataFrame(cv.cv_results_)[headers]
            results["cassifier"] = name
            df = pd.concat((df, results), sort=False)
        print(
            df.reindex(sorted(df.columns), axis=1).sort_values(
                "balanced_accuracy", ascending=False
            )
        )
