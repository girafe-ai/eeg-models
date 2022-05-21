from eeg_models.train_Demons import EEGtraining


sample_per_epoch = 400
sampling_rate = 512
decimation_factor = 5
filter = (4, 0.5, 20)
batch_size = 4
validation_split = 0.2
n_epochs = 2

a = EEGtraining(sample_per_epoch)

a.set_loaders(decimation_factor, sampling_rate, filter, batch_size, validation_split)

a.train_val(n_epochs)

decimator_pipeline = [1, 5]

filter_pipeline = [(4, 0.5, 20)]

a.searchgrid(
    decimator_pipeline,
    filter_pipeline,
    batch_size,
    validation_split,
    n_epochs,
    sampling_rate,
)
