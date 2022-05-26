from eeg_models.train_braininvaders import EegTraining


"""
    MODEL TRAINING & VALIDATION
    PARAMETER SEARCH
"""
sampling_rate = 512
epoch_duration = 0.9
decimation_factor = 10
filter = (4, 0.5, 20)
batch_size = 6
validation_split = 0.2
n_epochs = 2

# model training example
a = EegTraining(sampling_rate)
a.set_loaders(
    decimation_factor, epoch_duration, sampling_rate, filter, batch_size, validation_split
)
a.train_val(n_epochs)
