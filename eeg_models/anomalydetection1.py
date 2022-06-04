from typing import List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest

# from sklearn.metrics import silhouette_score
from sklearn.neighbors import LocalOutlierFactor


def outlier_remove(
    dataset,
    algo: str,
    train_list: List,
    subject_list: List,
):

    outlier_algorithm = ["IF", "LOF"]

    if algo not in set(outlier_algorithm):
        raise ValueError("algo must be 'IF' or 'LOF'")

    if algo == "IF":
        model = IsolationForest(
            n_estimators=100,
            max_samples="auto",
            contamination=0.05,
            max_features=1.0,
        )
        print(
            "Search of outlier epochs with IsolationForest, for subject  :", subject_list
        )

    elif algo == "LOF":
        model = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
        print(
            "Search of outlier epochs with LocalOutlierFactor, for subject  :",
            subject_list,
        )

    # signal_data is data for training outlier algorithm
    signal_data, all_label, nrows = outliers_training_data(train_list, dataset, algo)

    outliers_index_list = [[] for i in range(len(dataset))]

    # data & labels for subject if one subject
    if algo == "IF":
        outliers_index_list = sub_IF(
            model,
            dataset,
            subject_list,
            signal_data,
            all_label,
            nrows,
            outliers_index_list,
        )
    elif algo == "LOF":
        """
        LOF must be fitted and used to predict with the same data
        """
        outliers_index_list = sub_LOF(
            model,
            signal_data,
            subject_list,
            dataset,
            outliers_index_list,
            all_label,
            nrows,
        )
    # return for each subject in subject_list : a list of tuple (epoch, label) of outlier epochs
    return outliers_index_list


def sub_IF(
    model,
    dataset,
    subject_list,
    signal_data,
    all_label,
    nrows,
    outliers_index_list,
):

    # TRAINING with all data of all subject or with data of some subjects
    model.fit(signal_data)

    if subject_list == ["ALL"] or subject_list == ["all"]:
        subject_list = list(range(0, len(dataset)))

    if subject_list == [] or subject_list is None:
        raise (ValueError("subject_list is empty"))
    else:
        # subject data & labels
        for subject in subject_list:
            output = dataset[subject]

            output_data = output[0].numpy()
            output_label = output[1].numpy()
            outlier_count = {}
            for j in range(output_data.shape[0]):
                # flatten each epoch of the subject : 8 x n_samples of the current epoch
                print("subject :", subject, " - epoch : ", j)
                signal_data = output_data[j].flatten()
                pred = model.predict(signal_data.reshape(-1, signal_data.shape[0]))
                if pred == -1:
                    outliers_index_list[subject].append((j, output_label[j]))
                    outlier_count[j] = 1
            print(
                "for subject :",
                subject,
                "number of outlier epochs = ",
                len(outlier_count),
            )
            print(
                "outliers with isolation_forest (subject, epoch, label) : ",
                outliers_index_list,
            )

            plt.figure(figsize=(8, 6))
            plt.axis([0, nrows, 0, 1.2])
            x = np.linspace(0, nrows - 1, nrows)
            y = np.zeros((1, nrows))
            y[0, list(outlier_count)] = 1
            y = y.reshape(-1)
            plt.scatter(x, y, c="r")
            plt.show()

    return outliers_index_list


def sub_LOF(
    model,
    signal_data,
    subject_list,
    dataset,
    outliers_index_list,
    all_label,
    nrows,
):
    # Fit the model with all the data X  and predict for all data X
    outlier_labels = model.fit_predict(signal_data)

    # extract labels for current subject or for all subject
    if subject_list == ["ALL"] or subject_list == ["all"]:
        for j in range(len(outlier_labels)):
            if outlier_labels[j] == -1:
                subject_index, index_in_epoch = divmod(j, nrows)
                outliers_index_list[subject_index].append((index_in_epoch, all_label[j]))
        for j in range(len(dataset)):
            print(
                "for subject :",
                j,
                "number of outlier epochs = ",
                len(outliers_index_list[j]),
            )

    elif subject_list != [] and subject_list is not None:
        # subject_list has less elements than len(dataset)
        for subject in subject_list:
            outlier_labels_of_subject = outlier_labels[
                subject * nrows : (subject + 1) * nrows
            ]
            for j in range(len(outlier_labels_of_subject)):
                if outlier_labels_of_subject[j] == -1:
                    outliers_index_list[subject].append((j, all_label[j]))

        for subject in subject_list:
            print(
                "for subject :",
                subject,
                " - number of outlier epochs = ",
                len(outliers_index_list[subject]),
            )

    return outliers_index_list


def outliers_training_data(train_list, dataset, algo):

    all_data = []
    all_label = []

    if train_list[0] == "ALL" or algo == "LOF":
        train_list = list(range(0, len(dataset)))

    for i in range(len(train_list)):
        output = dataset[train_list[i]]
        output_0 = output[0].numpy()
        output_1 = output[1].numpy()
        print(train_list[i])
        for j in range(output_0.shape[0]):
            # flatten each epoch of the subject : 8 x n_samples of the current epoch
            all_data.append(output_0[j].flatten())
            # store label of each epoch
            all_label.append(output_1[j])
    nrows = output_0.shape[0]
    signal_data = np.vstack(all_data)

    return signal_data, all_label, nrows
