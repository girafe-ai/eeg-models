from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eeg_models.anomalydetection import clearSubjectDataset
from eeg_models.datasets.demons import DemonsP300Dataset
from eeg_models.train_Demons import EEGtraining
from eeg_models.transforms1 import (
    ButterFilter,
    ChannellwiseScaler,
    Decimator,
    MarkersTransformer,
)
from eeg_models.vizualisation import multiple_plot_data


"""
    MODEL TRAINING & VALIDATION
    PARAMETER SEARCH
"""
sample_per_epoch = 400
sampling_rate = 512
decimation_factor = 5
filter = (4, 0.5, 20)
batch_size = 6
validation_split = 0.2
n_epochs = 2

# model training example
a = EEGtraining(sample_per_epoch)
a.set_loaders(decimation_factor, sampling_rate, filter, batch_size, validation_split)
a.train_val(n_epochs)

# search of parameters :
decimator_pipeline = [1]
filter_pipeline = [(4, 0.5, 20)]
a.searchgrid(
    decimator_pipeline,
    filter_pipeline,
    batch_size,
    validation_split,
    n_epochs,
    sampling_rate,
)

"""
    PLOT
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
number_sample = 1200
# list of epoch index to plot
index = [0, 10, 100]
multiple_plot_data(my_dataset, subject_id, number_sample, index)

"""
    ANOMALY DETECTION
"""
# anomaly detection exemple
clearSubjectDataset(my_dataset, subject_id=0, session_id=0, run_id=0)
