import datetime
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eeg_models.anomalydetection1 import outlier_remove
from eeg_models.datasets.demons import DemonsP300Dataset
from eeg_models.train_Demons1 import filter_decim_searchgrid

# from eeg_models.train_Demons1 import (
#     EEGtraining,
#     filter_decim_searchgrid,
#     print_losses,
#     print_metric_results,
# )
from eeg_models.transforms1 import (
    ButterFilter,
    ChannellwiseScaler,
    Decimator,
    MarkersTransformer,
)
from eeg_models.vizualisation import multiple_plot_data


# from eeg_models.anomalydetection import clearSubjectDataset


"""
    OUTLIER REMOVAL
"""

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

sample_per_epoch = 300
index = 10

# list of subject indices used for outlier algorithm training
# index_list = ["ALL"]
index_list = [0, 10, 50]

# subject indice on which outlier detection is applied : indice or "ALL"
# subject = "ALL"
subject = 15

"""
    outlier algorithm & training & output
    - "IF" : Isolation Forest
    - "LOF" : Local Outlier Factor
"""

algorithm = "IF"

outliers = outlier_remove(
    algorithm, index_list, subject, eeg_pipe, markers_pipe, sample_per_epoch
)
print(outliers)


"""
    MODEL TRAINING & VALIDATION
    PARAMETER SEARCH
"""

# sample_per_epoch = 400
# sampling_rate = 512
# decimation_factor = 5
# filter = (4, 0.5, 20)
# batch_size = 4
# validation_split = 0.2
# n_epochs = 2

"""
    Model definition : nn_parameters
        n_classes: int,
        n_channels: int = 64,
        n_samples: int = 128,
        dropout_rate: float = 0.5,
        rate: int = 128,
        f1: int = 8,
        d: int = 2,
        f2: Optional[int] = None,

"""
# if not sample_per_epoch // decimation_factor:
#     raise ValueError("Decimation factor must divide sample_per_epoch")

# nn_parameters = {'n_classes' : 2, 'n_channels' : 8, 'n_samples' : sample_per_epoch // decimation_factor, 'dropout_rate' : 0.5, \
#      'rate' : 128, 'f1' : 8, 'd' : 2, 'f2' : None}


"""
    MODEL TRAINING & VALIDATION
"""

# model_to_train = EEGtraining(nn_parameters, sample_per_epoch)
# model_to_train.set_loaders(decimation_factor, sampling_rate, filter, batch_size, validation_split)
# metrics_model, train_losses, val_losses = model_to_train.train_val(n_epochs)
# print_metric_results(metric_results)
# print_losses(train_losses, val_losses)


"""
    SEARCH OF PARAMETERS

    MODEL PARAMETER SEARCH

    don't change :
    - n_classes = 2  (output class number)
    - n_channels = 8 (input channel number)

"""

sampling_rate = 512
batch_size = 4
validation_split = 0.2
n_epochs = 20


nn_parameters_pipeline = [
    {
        "n_classes": 2,
        "n_channels": 8,
        "n_samples": 400,
        "dropout_rate": 0.5,
        "rate": 254,
        "f1": 8,
        "d": 2,
        "f2": None,
    },  # {'n_classes' : 2, 'n_channels' : 8, 'n_samples' : 400, 'dropout_rate' : 0.5, 'rate' : 128, 'f1' : 8, 'd' : 2, 'f2' : None}, \
    # {'n_classes' : 2, 'n_channels' : 8, 'n_samples' : 400, 'dropout_rate' : 0.2, 'rate' : 128, 'f1' : 8, 'd' : 2, 'f2' : None}
]

sample_per_epoch_pipeline = [
    400,
    # 600
]

decimator_pipeline = [
    5,
    # 5
]


filter_pipeline = [
    (4, 0.5, 20),
    # (5, 0.5, 20)
]


results = filter_decim_searchgrid(
    nn_parameters_pipeline,
    sample_per_epoch_pipeline,
    decimator_pipeline,
    filter_pipeline,
    batch_size,
    validation_split,
    n_epochs,
    sampling_rate,
)


"""
    PLOT results
    transform results in dataframe
    results : nn_parameters, sample_per_epoch, decimation_factor, filter, metrics_param_model, train_losses, val_losses)
"""

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


"""
    SAVE RESULTS
"""

# import datetime
# import pickle


date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# with open('test.pickle', 'rb') as infile : results = pickle.load(infile)

with open(f"results-{date}.pickle", "wb") as outfile:
    pickle.dump(results, outfile)

# with open(f'results-{date}', 'rb') as infile : results = pickle.load(infile)


# result_test.iloc[0]['A'] : pour accéder à l'élément de la ligne 0 et colonne A
# result_test.shape[0]  : nombre de lignes

row, col = results.shape

# for i in range(row):
for i in range(row):
    plt.figure(figsize=(10, 10))
    plt.title(
        f'{results.iloc[i]["sample_per_epoch"]} : samples per epoch  - {results.iloc[i]["decimation_factor"]} : decimation factor - {results.iloc[i]["filter"]} : filter'
    )
    plt.plot(results.iloc[i]["train_losses"], label="train")
    plt.plot(results.iloc[i]["val_losses"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()

# for i in range(row):
for i in range(row):
    plt.figure(figsize=(10, 10))
    y = []
    for j in range(len(results.iloc[i]["metrics_param_model"])):
        y.append(results.iloc[i]["metrics_param_model"][j][3])
    plt.title(
        f'{results.iloc[i]["sample_per_epoch"]} : samples per epoch  - {results.iloc[i]["decimation_factor"]} : decimation factor - {results.iloc[i]["filter"]} : filter'
    )
    plt.xlabel("Epochs")
    plt.ylabel("(val) f1 score")
    plt.plot(np.array(y), label="f1 score")
    plt.legend()
    plt.show()


# for i in range(row):
for i in range(row):
    plt.figure(figsize=(10, 10))
    y = []
    for j in range(len(results.iloc[i]["metrics_param_model"])):
        y.append(results.iloc[i]["metrics_param_model"][j][1])
    plt.title(
        f'{results.iloc[i]["sample_per_epoch"]} : samples per epoch  - {results.iloc[i]["decimation_factor"]} : decimation factor - {results.iloc[i]["filter"]} : filter'
    )
    plt.xlabel("Epochs")
    plt.ylabel("(val) accuracy")
    plt.plot(np.array(y), label="accuracy")
    plt.legend()
    plt.show()


"""
    PLOT :  plot of runs (by index) of the same subject
"""
# sampling_rate
sampling_rate = 500
decimation_factor = 1
order = 4
highpass = 0.5
lowpass = 20
labels_mapping = {1: 1, 2: 0, 0: 0}
# sample per epoch of each epoch output from getitem
sample_per_epoch = 400
eeg_pipe = make_pipeline(
    Decimator(decimation_factor),
    ButterFilter(sampling_rate // decimation_factor, order, highpass, lowpass),
    ChannellwiseScaler(StandardScaler()),
)
markers_pipe = MarkersTransformer(labels_mapping, decimation_factor)
# dataset init
n_samplesdecimated = sample_per_epoch // decimation_factor
my_dataset = DemonsP300Dataset(
    transform=eeg_pipe, target_transform=markers_pipe, sample_per_epoch=sample_per_epoch
)
# PLOT  : index is here index of subject
# plot data for subject 0 (17 epochs)
#    index of subject in [0, ..., 60]
subject_id = 0
#  max number of samples to plot
number_sample = 400
# list of epoch index to plot
index = [0, 11, 22]
multiple_plot_data(my_dataset, subject_id, number_sample, index)

"""
    ANOMALY DETECTION
"""
# anomaly detection exemple
# clearSubjectDataset(my_dataset, subject_id=0, session_id=0, run_id=0)
