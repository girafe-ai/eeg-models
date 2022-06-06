# from eeg_models.train_braininvaders import EegTraining


# sampling_rate = 512
# epoch_duration = 0.9
# decimation_factor = 10
# filter = (4, 0.5, 20)
# batch_size = 6
# validation_split = 0.2
# n_epochs = 2

# train_braininvaders= EegTraining(sampling_rate)
# train_braininvaders.set_loaders(
#     decimation_factor, epoch_duration, sampling_rate, filter, batch_size, validation_split
# )
# train_braininvaders.train_val(n_epochs)


from eeg_models.train_val_test import EegTraining


sampling_rate = 512
epoch_duration = 0.9
decimation_factor = 10
filter = (4, 0.5, 20)
batch_size = 10
validation_split = 0.2
n_epochs = 20

train_braininvaders = EegTraining(sampling_rate)
train_braininvaders.set_loaders(
    decimation_factor, epoch_duration, sampling_rate, filter, batch_size, validation_split
)
train_braininvaders.train_val_test(n_epochs)
