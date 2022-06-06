import datetime
import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from somepytools.typing import File

# from eeg_models.anomalydetection1 import outlier_remove
from eeg_models.datasets.demons2 import DemonsP300Dataset
from eeg_models.train_Demons3 import filter_decim_searchgrid
from eeg_models.transforms1 import (
    ButterFilter,
    ChannellwiseScaler,
    Decimator,
    MarkersTransformer,
)


# from somepytools.typing import (
#     Any,
#     Array,
#     Callable,
#     Dict,
#     Directory,
#     File,
#     Optional,
#     Sequence,
#     Tuple,
# )


"""
    OUTLIER REMOVAL
"""

# dataset for outlier detection

labels_mapping = {1: 1, 2: 0, 0: 0}
decimation_factor = 1
sampling_rate = 512
order, highpass, lowpass = (4, 0.5, 20)

eeg_pipe = make_pipeline(
    Decimator(decimation_factor),
    ButterFilter(sampling_rate // decimation_factor, order, highpass, lowpass),
    ChannellwiseScaler(StandardScaler()),
)

markers_pipe = MarkersTransformer(labels_mapping, decimation_factor)

#  window size (300 or 400 or 500 ...) -> should be the same for outlier detection and classification
sample_per_epoch = 400

dataset = DemonsP300Dataset(
    transform=eeg_pipe,
    target_transform=markers_pipe,
    sample_per_epoch=sample_per_epoch,
    use_cache=True,
)

print(dataset[1])

print(dataset[1])

"""
    outlier algorithm & training & output
    - "IF" : Isolation Forest
    - "LOF" : Local Outlier Factor
"""

# Algorithm to use for outlier detection
# algorithm = "LOF"
algorithm = "LOF"

# list of subject indices used for outlier algorithm training :  "ALL" or list of indices
train_list = ["ALL"]
# train_list = [1, 2, 3, 4, 5 ]

# computation of outliers for a list of subjects indices or "ALL"
subject_list = ["ALL"]
# subject_list = [5, 15, 35, 55]

# list of anomalies for subject in list of subject indices
# output outliers= list of list of tuples (index_of_epoch, labels_of_epoch)
# list of outliers tuples for 'i' subject in list of position = i


# outliers = outlier_remove(
#     dataset,
#     algorithm,
#     train_list,
#     subject_list,
# )


# show outliers for subject : subject_tmp = 10
# print("outlier epochs for subject : ", subject_tmp, " = ", [ outliers[subject_tmp][j][0] for j in range(len(outliers[subject_tmp]))] )


"""
    SAVE OUTLIERS  : dump  outliers in pickle file
"""

date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# save_result = True

# if save_result:
#     date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     file_name = (
#         f"outliers-{date}-algo-{algorithm}-sub-{subject_list}-train-{train_list}.pickle"
#     )
#     with open(file_name, "wb") as outfile:
#         pickle.dump(outliers, outfile)


"""
    LOAD  PREVIOUS RESULTS  (from pickle file : date, algo : IF/LOF, sub : ALL/list of subject indices, train : ALL/list of subject indices)
"""
file_name = "outliers-2022-06-06_00-11-46-algo-LOF-sub-['ALL']-train-['ALL'].pickle"
file_name_restore = file_name
with open(file_name_restore, "rb") as infile:
    outliers_restored = pickle.load(infile)

# check outliers for subject : subject_tmp = 10
# outliers_restored is a list of lists[tuple(indice of epoch, label of epoch)] = list of outliers per subject
# outliers_restored[i] is the list of outliers for subject i

# outliers_restored = outliers

print(outliers_restored)
# print("outlier epochs for subject : ", subject_tmp, " = ", [ outliers_restored[subject_tmp][j][0] for j in range(len(outliers_restored[subject_tmp]))] )

"""
    transform outliers into a dict of dicts
    dict of subjects with outliers
    for each subject : dict of (epoch, label)
"""
dict_of_subject_outliers = {}
for i in range(len(outliers_restored)):
    if len(outliers_restored[i]) > 0:
        dict_of_subject_outliers[i] = {
            outliers_restored[i][j][0]: int(outliers_restored[i][j][1])
            for j in range(len(outliers_restored[i]))
        }

"""
    SAVE IN JSON FILE
"""


def save_outliers(outliers: dict, filename: File) -> None:
    with open(filename, "w") as f:
        json.dump(outliers, f)


def load_outliers(filename: File) -> dict:
    with open(filename) as f:
        outliers = json.load(f)
    return outliers


dict_dump = f"outliers-{date}-algo-{algorithm}-sub-{subject_list}-train-{train_list}.json"
save_outliers(dict_of_subject_outliers, dict_dump)

# output = dict of dicts
# dict of subject with outliers : dict_of_subject_outliers
# for each subject : list of outlier epochs with label
# example :  { '11' :{ '0' : '13' :0, '20' : 1} }
dict_outliers_recovery = load_outliers(dict_dump)

print(dict_outliers_recovery == dict_of_subject_outliers)

outliers_filename = dict_dump


"""
    MODEL TRAINING & VALIDATION & PARAMETER SEARCH
"""


"""
    Reminder :

    Model definition in nn_parameters :
        n_classes: int,
        n_channels: int = 64,
        n_samples: int = 128,
        dropout_rate: float = 0.5,
        rate: int = 128,
        f1: int = 8,
        d: int = 2,
        f2: Optional[int] = None,

"""


"""
    MODEL TRAINING & VALIDATION  : test
"""

# if not sample_per_epoch // decimation_factor:
#     raise ValueError("Decimation factor must divide sample_per_epoch")
#
# define : filter, batch_size, validation_split, sampling_rate, decimataion_factor, nn_parameters, n_epochs
#
# sample_per_epoch = 400
# decimation_factor = 1
# nn_parameters = {'n_classes' : 2, 'n_channels' : 8, 'n_samples' : sample_per_epoch // decimation_factor, 'dropout_rate' : 0.5, \
#      'rate' : 128, 'f1' : 8, 'd' : 2, 'f2' : None}
#
# model_to_train = EEGtraining(nn_parameters, sample_per_epoch)
# model_to_train.set_loaders(decimation_factor, sampling_rate, filter, batch_size, validation_split)
# metrics_model, train_losses, val_losses = model_to_train.train_val(n_epochs)
# print_metric_results(metric_results)
# print_losses(train_losses, val_losses)


"""
    SEARCH OF PARAMETERS :
    define set of parameters fo nn_parameters, sample_per_epoch, decimation_factor, filter parameters
    searchgrid : training & val on all parameter combinations

    don't change :
    - n_classes = 2  (output class number)
    - n_channels = 8 (input channel number)  depend on dataset (in Demons : 8 channels are recorded)

"""
# reference window size
# can be change in sample_per_epoch_pipeline list : this last value will be used to reshape NN
# to keep it compatible with dataset window  (n_sample = y axis of data epochs)
NSAMPLES = 400

# check sampling rate is the same in dataset
sampling_rate = 512

# training and validation parameters
batch_size = 4
validation_split = 0.2
n_epochs = 60

# list of NN parameters to test
nn_parameters_pipeline = [
    # {
    #     "n_classes": 2,
    #     "n_channels": 8,
    #     "n_samples": NSAMPLES,
    #     "dropout_rate": 0.5,
    #     "rate": 128,
    #     "f1": 8,
    #     "d": 2,
    #     "f2": None,
    # },
    {
        "n_classes": 2,
        "n_channels": 8,
        "n_samples": NSAMPLES,
        "dropout_rate": 0.3,
        "rate": 128,
        "f1": 8,
        "d": 2,
        "f2": None,
    },
    # {'n_classes' : 2, 'n_channels' : 8, 'n_samples' : 400, 'dropout_rate' : 0.2, 'rate' : 128, 'f1' : 8, 'd' : 2, 'f2' : None}
]

# lis of window sizes to test
sample_per_epoch_pipeline = [
    # 300,
    # 400,
    500
]

# list of decimation factor to test
decimator_pipeline = [
    1,
    # 5,
]

# list of filter parameters to test
filter_pipeline = [
    (4, 0.5, 20),
    # (5, 0.5, 20)
]

# to use dataset without outliers epochs :  outliers_restored  must be not None, and equal to output of outlier_remove() function)
# outliers_restored = None
print(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

results = filter_decim_searchgrid(
    nn_parameters_pipeline,
    sample_per_epoch_pipeline,
    decimator_pipeline,
    filter_pipeline,
    batch_size,
    validation_split,
    n_epochs,
    sampling_rate,
    outliers_restored,
    outliers_filename,
)

print(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

"""
    PLOT results :
    - transform results in dataframe
    - results : nn_parameters, sample_per_epoch, decimation_factor, filter, metrics_param_model, train_losses, val_losses)
"""


# necessary for down computings : (dump, etc..)
results = pd.DataFrame(results)

results.columns = [
    "nn_parameters",
    "sample_per_epoch",
    "decimation_factor",
    "filter",
    "metrics_param_model",
    "train_losses",
    "val_losses",
]

row, col = results.shape

"""
    USE PREVIOUS RESULTS : load results from pickle file
"""
# with open(f"results-2022-06-04_02-30-17-algo-IF-sub-['ALL']-train-['ALL']-num_tests-4.pickle", 'rb') as infile:
#     results = pickle.load(infile)

"""
    SAVE RESULTS  : dump results in pickle file
"""
date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

with open(
    f"results-{date}-algo:{algorithm}-sub:{subject_list}-train:{train_list}-tests:{results.shape[0]}.pickle",
    "wb",
) as outfile:
    pickle.dump(results, outfile)


"""
    PLOTS : LOSS, F1, ACCURACY
"""

# PLOT : LOSS
for i in range(row):
    fig = plt.figure(figsize=(10, 10))
    if outliers_restored is None:
        plt.title(
            f"{results.iloc[i]['sample_per_epoch']}:s_per_epoch-{results.iloc[i]['decimation_factor']}:dec-{results.iloc[i]['filter']}:fi-{results.iloc[i]['nn_parameters']['rate']}:rate-{results.iloc[i]['nn_parameters']['dropout_rate']}:drop"
        )
    else:
        plt.title(
            f"{algorithm}:algo-{results.iloc[i]['sample_per_epoch']}:sa_per_epoch-{results.iloc[i]['decimation_factor']}:dec-{results.iloc[i]['filter']}:fi-{results.iloc[i]['nn_parameters']['rate']}:rate-{results.iloc[i]['nn_parameters']['dropout_rate']}:drop"
        )

    plt.plot(results.iloc[i]["train_losses"], label="train")
    plt.plot(results.iloc[i]["val_losses"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()
    if outliers_restored is None:
        fig.savefig(
            f"loss-{date}-test:{i}-rate:{results.iloc[i]['nn_parameters']['rate']}-drop:{results.iloc[i]['nn_parameters']['dropout_rate']}-dec:{results.iloc[i]['decimation_factor']}-spe:{results.iloc[i]['sample_per_epoch']}-fil:{results.iloc[i]['filter']}.png"
        )
    else:
        fig.savefig(
            f"loss-{date}-algo:{algorithm}-sub:{subject_list}-train:{train_list}-test:{i}-rate:{results.iloc[i]['nn_parameters']['rate']}-drop:{results.iloc[i]['nn_parameters']['dropout_rate']}-dec:{results.iloc[i]['decimation_factor']}-spe:{results.iloc[i]['sample_per_epoch']}-fil:{results.iloc[i]['filter']}.png"
        )

# PLOT : F1
for i in range(row):
    fig = plt.figure(figsize=(10, 10))
    y = []
    for j in range(len(results.iloc[i]["metrics_param_model"])):
        y.append(results.iloc[i]["metrics_param_model"][j][4])
    if outliers_restored is None:
        plt.title(
            f"{results.iloc[i]['sample_per_epoch']}:s_per_epoch-{results.iloc[i]['decimation_factor']}:dec-{results.iloc[i]['filter']}:fi-{results.iloc[i]['nn_parameters']['rate']}:rate-{results.iloc[i]['nn_parameters']['dropout_rate']}:drop"
        )
    else:
        plt.title(
            f"{algorithm}:algo-{results.iloc[i]['sample_per_epoch']}:sa_per_epoch-{results.iloc[i]['decimation_factor']}:dec-{results.iloc[i]['filter']}:fi-{results.iloc[i]['nn_parameters']['rate']}:rate-{results.iloc[i]['nn_parameters']['dropout_rate']}:drop"
        )
    plt.xlabel("Epochs")
    plt.ylabel("(val) f1 score")
    plt.plot(np.array(y), label="f1 score")
    plt.legend()
    plt.show()
    if outliers_restored is None:
        fig.savefig(
            f"f1-{date}-test:{i}-rate:{results.iloc[i]['nn_parameters']['rate']}-drop:{results.iloc[i]['nn_parameters']['dropout_rate']}-dec:{results.iloc[i]['decimation_factor']}-spe:{results.iloc[i]['sample_per_epoch']}-fil:{results.iloc[i]['filter']}.png"
        )
    else:
        fig.savefig(
            f"f1-{date}-algo:{algorithm}-sub:{subject_list}-train:{train_list}-test:{i}-rate:{results.iloc[i]['nn_parameters']['rate']}-drop:{results.iloc[i]['nn_parameters']['dropout_rate']}-dec:{results.iloc[i]['decimation_factor']}-spe:{results.iloc[i]['sample_per_epoch']}-fil:{results.iloc[i]['filter']}.png"
        )


#  PLOT : ACCURACY
for i in range(row):
    fig = plt.figure(figsize=(10, 10))
    y = []
    for j in range(len(results.iloc[i]["metrics_param_model"])):
        y.append(results.iloc[i]["metrics_param_model"][j][1])
    if outliers_restored is None:
        plt.title(
            f"{results.iloc[i]['sample_per_epoch']}:s_per_epoch-{results.iloc[i]['decimation_factor']}:dec-{results.iloc[i]['filter']}:fi-{results.iloc[i]['nn_parameters']['rate']}:rate-{results.iloc[i]['nn_parameters']['dropout_rate']}:drop"
        )
    else:
        plt.title(
            f"{algorithm}:algo-{results.iloc[i]['sample_per_epoch']}:sa_per_epoch-{results.iloc[i]['decimation_factor']}:dec-{results.iloc[i]['filter']}:fi-{results.iloc[i]['nn_parameters']['rate']}:rate-{results.iloc[i]['nn_parameters']['dropout_rate']}:drop"
        )
    plt.xlabel("Epochs")
    plt.ylabel("(val) accuracy")
    plt.plot(np.array(y), label="accuracy")
    plt.legend()
    plt.show()
    if outliers_restored is None:
        fig.savefig(
            f"accuracy-{date}-test:{i}-rate:{results.iloc[i]['nn_parameters']['rate']}-drop:{results.iloc[i]['nn_parameters']['dropout_rate']}-dec:{results.iloc[i]['decimation_factor']}-spe:{results.iloc[i]['sample_per_epoch']}-fil:{results.iloc[i]['filter']}.png"
        )
    else:
        fig.savefig(
            f"accuracy-{date}-algo:{algorithm}-sub:{subject_list}-train:{train_list}-test:{i}-rate:{results.iloc[i]['nn_parameters']['rate']}-drop:{results.iloc[i]['nn_parameters']['dropout_rate']}-dec:{results.iloc[i]['decimation_factor']}-spe:{results.iloc[i]['sample_per_epoch']}-fil:{results.iloc[i]['filter']}.png"
        )
