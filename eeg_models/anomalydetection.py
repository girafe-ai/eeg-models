import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.neighbors import LocalOutlierFactor


def clearSubjectDataset(
    dataset, subject_id: int = 0, session_id: int = 0, run_id: int = 0
):
    """
    function to clear outliers on one subject data
    output : list of index of samples to remove from subject dataset
    identified as outliers

    """
    record = dataset._read_hdf(dataset.root / dataset.meta.loc[subject_id, "filename"])
    runs_raw = {}
    for i, act in enumerate(record):
        # target and stims are increased by 1
        # because the channel is filled with zeros by default
        target = act["target"] + 1
        run_data = []
        for eeg, starts, stims in act["sessions"]:
            starts = starts * dataset.sampling_rate / dataset._ms_in_sec
            starts = starts.round().astype(np.int)
            stims = stims + 1
            stims_channel = np.zeros(eeg.shape[1])
            target_channel = np.zeros(eeg.shape[1])

            for start, stimul in zip(starts, stims):
                stims_channel[start] = stimul
                target_channel[start] = 1 if stimul == target else 2

            round_data = np.vstack((eeg, stims_channel[None, :], target_channel[None, :]))
            run_data.append(round_data)
        raw = np.hstack(run_data)
        runs_raw[f"run_{i}"] = raw
    data = {"session_0": runs_raw}
    signal_data = data[f"session_{session_id}"][f"run_{run_id}"][0:8]
    df = pd.DataFrame(signal_data)
    df2 = df.T

    # anomaly detection with IsolationForest
    model = IsolationForest(
        n_estimators=100,
        max_samples="auto",
        contamination="auto",
        max_features=1.0,
    )
    model.fit(df2)
    pred = model.predict(df2)
    df2["anomaly_if"] = pred
    outlier_if = df2.loc[df2["anomaly_if"] == -1]
    outlier_if_index = list(outlier_if.index)
    print("IsolationForest - len(outlier_if_index) = ", len(outlier_if_index))
    print("IsolationForest - outlier_if_index : ", outlier_if_index)
    df2.drop(["anomaly_if"], axis=1, inplace=True)
    print(
        "IsolationForest - silhouette_score isolation forest : ",
        silhouette_score(df2, pred),
    )

    # anomaly detection with LocalOutlierFactor
    lof = LocalOutlierFactor(n_neighbors=20, contamination="auto")
    y_pred = lof.fit_predict(df2)
    df2["anomaly_lof"] = y_pred
    outlier_lof = df2.loc[df2["anomaly_lof"] == -1]
    outlier_lof_index = list(outlier_lof.index)
    print("LocalOutlierFactor - len(outlier_lof_index) = ", len(outlier_lof_index))
    print("LocalOutlierFactor - outlier_lof_index: ", outlier_lof_index)
    df2.drop(["anomaly_lof"], axis=1, inplace=True)
    print(
        "LocalOutlierFactor - silhouette_score LocalOutlierFactor : ",
        silhouette_score(df2, y_pred),
    )

    # Keep algorithm LocalOutlierFactor - to do :  search grid for parameters and algorithm wrt model performance

    # remove outliers from df2 and reconstruct original numpy array
    df3 = df2.drop(outlier_lof_index)
    df_result = df3.T
    np_df_result = df_result.to_numpy()
    print("original shape with outliers : ", signal_data.shape)
    print("result shape without outliers :", np_df_result.shape)
