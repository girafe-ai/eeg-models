from eeg_models.train_pyrimanian import TrainPyRimanianClassifier


decimation_factor = 11
epoch_duration = 0.9
sampling_rate = 512
filter = (4, 0.5, 20)

train = TrainPyRimanianClassifier()

train.set_dataset(decimation_factor, epoch_duration, sampling_rate, filter)

train.crossvalidate_record()
