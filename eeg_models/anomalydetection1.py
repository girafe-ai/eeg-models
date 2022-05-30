from collections import Counter
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest

# from sklearn.metrics import silhouette_score
from sklearn.neighbors import LocalOutlierFactor

from eeg_models.datasets.demons import DemonsP300Dataset


def outlier_remove(
    algo: str,
    index_list: List,
    subject: int,
    eeg_pipe,
    markers_pipe,
    sample_per_epoch,
):

    outlier_algorithm = ["IF", "LOF"]

    if algo not in set(outlier_algorithm):
        raise ValueError("algo must be 'IF' or 'LOF'")

    dataset = DemonsP300Dataset(
        transform=eeg_pipe,
        target_transform=markers_pipe,
        sample_per_epoch=sample_per_epoch,
    )

    model_if = IsolationForest(
        n_estimators=100,
        max_samples="auto",
        contamination=0.05,
        max_features=1.0,
    )

    model_lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)

    #  training data for outlier detection algorithm

    all_data = []
    all_label = []

    if index_list[0] == "ALL" or algo == "LOF":
        index_list = list(range(0, len(dataset)))

    for i in range(len(index_list)):
        output = dataset[index_list[i]]
        output_0 = output[0].numpy()
        output_1 = output[1].numpy()
        print(index_list[i])
        for j in range(output_0.shape[0]):
            # flatten each epoch of the subject : 8 x n_samples of the current epoch
            all_data.append(output_0[j].flatten())
            # store label of each epoch
            all_label.append(output_1[j])

    nrows = output_0.shape[0]

    signal_data = np.vstack(all_data)

    if algo == "IF":
        model = model_if
        print("Search of outlier epochs with IsolationForest, for subject  :", subject)
    elif algo == "LOF":
        model = model_lof
        print("Search of outlier epochs with LocalOutlierFactor, for subject  :", subject)

    outliers_index_list = []

    # number of subjects in dataset
    # n_subject = len(dataset)

    # data & labels for subject if one subject
    if subject != "ALL":
        # subject data & labels
        output = dataset[subject]
        # number of epochs in subject
        nrows = (output[0].numpy()).shape[0]
        # data & labels of one subject
        output_data = output[0].numpy()
        output_label = output[1].numpy()

    if algo == "IF":

        outliers_index_list = sub_IF(
            model,
            signal_data,
            subject,
            output_data,
            output_label,
            nrows,
            outliers_index_list,
            all_label,
        )

        # # TRAINING with all data of all subject or with data of some subjects
        # model.fit(signal_data)
        # if subject != "ALL":
        #     for j in range(output_data.shape[0]):
        #         # flatten each epoch of the subject : 8 x n_samples of the current epoch
        #         print("subject :", subject, " - epoch : ", j)
        #         signal_data = output_data[j].flatten()
        #         pred = model.predict(signal_data.reshape(-1, signal_data.shape[0]))
        #         # remove outlier if pred = -1
        #         if pred == -1:
        #             outliers_index_list.append((subject, j, output_label[j]))
        #     print(
        #         "for subject :",
        #         subject,
        #         "number of outlier epochs = ",
        #         len(outliers_index_list),
        #     )
        #     print(
        #         "outliers with isolation_forest (subject, epoch, label) : ",
        #         outliers_index_list,
        #     )

        #     plt.figure(figsize=(8, 6))
        #     plt.axis([0, nrows, 0, 1.2])
        #     x = np.linspace(0, nrows - 1, nrows)
        #     y = np.zeros((1, nrows))
        #     y[0, [outliers_index_list[i][1] for i in range(len(outliers_index_list))]] = 1
        #     y = y.reshape(-1)
        #     plt.scatter(x, y, c="r")
        #     plt.show()
        # else:
        #     outlier_labels = model.predict(signal_data)
        #     count = Counter()
        #     for j in range(len(outlier_labels)):
        #         if outlier_labels[j] == -1:
        #             subject_index, index_in_epoch = divmod(j, nrows)
        #             outliers_index_list.append(
        #                 (subject_index, index_in_epoch, all_label[j])
        #             )
        #     count.update(outlier_labels)
        #     print("for subject :", subject, " - number of outlier epochs = ", count[(-1)])
        #     print(
        #         " liste of outliers with IF (subject, epoch, label) : \n",
        #         outliers_index_list,
        #     )

    elif algo == "LOF":

        outliers_index_list = sub_LOF(
            model,
            signal_data,
            subject,
            outliers_index_list,
            all_label,
            nrows,
            output_label,
        )

        # # Fit the model to the training set X and return the labels
        # outlier_labels = model.fit_predict(signal_data)
        # count = Counter()
        # # extract labels for current subject or for all subject
        # if subject == "ALL":
        #     for j in range(len(outlier_labels)):
        #         if outlier_labels[j] == -1:
        #             subject_index, index_in_epoch = divmod(j, nrows)
        #             outliers_index_list.append(
        #                 (subject_index, index_in_epoch, all_label[j])
        #             )
        #     count.update(outlier_labels)
        #     print("for subject :", subject, " - number of outlier epochs = ", count[(-1)])
        #     print(
        #         " liste of outliers with LOF (subject, epoch, label) : \n",
        #         outliers_index_list,
        #     )

        # else:
        #     outlier_labels_of_subject = outlier_labels[
        #         subject * nrows : (subject + 1) * nrows
        #     ]
        #     for j in range(len(outlier_labels_of_subject)):
        #         if outlier_labels_of_subject[j] == -1:
        #             outliers_index_list.append((subject, j, output_label[j]))

        #     count.update(outlier_labels_of_subject)
        #     print("for subject :", subject, " - number of outlier epochs = ", count[(-1)])
        #     print(
        #         " liste of outliers with LOF (subject, epoch, label) : \n",
        #         outliers_index_list,
        #     )

    return outliers_index_list


def sub_IF(
    model,
    signal_data,
    subject,
    output_data,
    output_label,
    nrows,
    outliers_index_list,
    all_label,
):
    # TRAINING with all data of all subject or with data of some subjects
    model.fit(signal_data)
    if subject != "ALL":
        for j in range(output_data.shape[0]):
            # flatten each epoch of the subject : 8 x n_samples of the current epoch
            print("subject :", subject, " - epoch : ", j)
            signal_data = output_data[j].flatten()
            pred = model.predict(signal_data.reshape(-1, signal_data.shape[0]))
            # remove outlier if pred = -1
            if pred == -1:
                outliers_index_list.append((subject, j, output_label[j]))
        print(
            "for subject :",
            subject,
            "number of outlier epochs = ",
            len(outliers_index_list),
        )
        print(
            "outliers with isolation_forest (subject, epoch, label) : ",
            outliers_index_list,
        )

        plt.figure(figsize=(8, 6))
        plt.axis([0, nrows, 0, 1.2])
        x = np.linspace(0, nrows - 1, nrows)
        y = np.zeros((1, nrows))
        y[0, [outliers_index_list[i][1] for i in range(len(outliers_index_list))]] = 1
        y = y.reshape(-1)
        plt.scatter(x, y, c="r")
        plt.show()
    else:
        outlier_labels = model.predict(signal_data)
        count = Counter()
        for j in range(len(outlier_labels)):
            if outlier_labels[j] == -1:
                subject_index, index_in_epoch = divmod(j, nrows)
                outliers_index_list.append((subject_index, index_in_epoch, all_label[j]))
        count.update(outlier_labels)
        print("for subject :", subject, " - number of outlier epochs = ", count[(-1)])
        print(
            " liste of outliers with IF (subject, epoch, label) : \n",
            outliers_index_list,
        )

    return outliers_index_list


def sub_LOF(
    model, signal_data, subject, outliers_index_list, all_label, nrows, output_label
):
    # Fit the model to the training set X and return the labels
    outlier_labels = model.fit_predict(signal_data)
    count = Counter()
    # extract labels for current subject or for all subject
    if subject == "ALL":
        for j in range(len(outlier_labels)):
            if outlier_labels[j] == -1:
                subject_index, index_in_epoch = divmod(j, nrows)
                outliers_index_list.append((subject_index, index_in_epoch, all_label[j]))
        count.update(outlier_labels)
        print("for subject :", subject, " - number of outlier epochs = ", count[(-1)])
        print(
            " liste of outliers with LOF (subject, epoch, label) : \n",
            outliers_index_list,
        )

    else:
        outlier_labels_of_subject = outlier_labels[
            subject * nrows : (subject + 1) * nrows
        ]
        for j in range(len(outlier_labels_of_subject)):
            if outlier_labels_of_subject[j] == -1:
                outliers_index_list.append((subject, j, output_label[j]))

        count.update(outlier_labels_of_subject)
        print("for subject :", subject, " - number of outlier epochs = ", count[(-1)])
        print(
            " liste of outliers with LOF (subject, epoch, label) : \n",
            outliers_index_list,
        )

    return outliers_index_list
