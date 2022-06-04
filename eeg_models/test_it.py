import numpy as np
from matplotlib import pyplot as plt

from eeg_models.train_val_test import EegTraining


# from eeg_models.train_braininvaders import EegTraining


# sampling_rate = 512
# epoch_duration = 0.9
# decimation_factor = 10
# filter = (4, 0.5, 20)
# batch_size = 20
# validation_split = 0.2
# n_epochs = 120

# train_braininvaders = EegTraining(sampling_rate)
# train_braininvaders.set_loaders(
#     decimation_factor, epoch_duration, sampling_rate, filter, batch_size, validation_split
# )
# train_braininvaders.train_val(n_epochs)


# from eeg_models.train_val_test import EegTraining


# sampling_rate = 512
# decimation_factor = 10
# epoch_duration = 0.9
# filter = (4, 0.5, 20)
# batch_size = 10
# validation_split = 0.2
# n_epochs = 20

# train_braininvaders = EegTraining(sampling_rate)
# train_braininvaders.set_loaders(
#     decimation_factor, epoch_duration, sampling_rate, filter, batch_size, validation_split
# )
# train_braininvaders.train_val_test(n_epochs)

# sampling_rate = 512
# decimator_pipeline = [9, 10, 11]
# epoch_duration_pipeline = [0.8, 0.9, 1.0]
# filter_pipeline = [(4, 0.5, 20), (4, 0.4, 20), (3, 0.6, 18), (5, 0.5, 20)]
# batch_size = 10
# validation_split = 0.2
# n_epochs = 10
# train_braininvaders = EegTraining(sampling_rate)

# train_braininvaders.searchgrid(
#     decimator_pipeline,
#     epoch_duration_pipeline,
#     filter_pipeline,
#     batch_size,
#     validation_split,
#     n_epochs,
#     sampling_rate,
# )

# print(train_braininvaders.results)


parameters = []
balanced_test_accuracy = []

decimation_factors = [8]
epoch_durations = [0.8, 0.9, 1.0]
orders = [3, 4, 5]
highpasses = [0.4, 0.5, 0.6]
lowpasses = [19, 20, 21]

for d in decimation_factors:
    for e in epoch_durations:
        for o in orders:
            for h in highpasses:
                for low in lowpasses:
                    parameters.append((d, e, o, h, low))


for i in range(len(parameters)):
    sampling_rate = 512
    decimation_factor = parameters[i][0]
    epoch_duration = parameters[i][1]
    filter = (parameters[i][2], parameters[i][3], parameters[i][4])
    batch_size = 20
    validation_split = 0.2
    n_epochs = 10

    train_braininvaders = EegTraining(sampling_rate)
    train_braininvaders.set_loaders(
        decimation_factor,
        epoch_duration,
        sampling_rate,
        filter,
        batch_size,
        validation_split,
    )
    train_braininvaders.train_val_test(n_epochs)
    balanced_test_accuracy.append(train_braininvaders.results)

x = [p for p in range(len(parameters))]


plt.figure(figsize=(15, 15))
plt.plot(x, balanced_test_accuracy, marker=".", label="balanced_test_accuracy")
plt.title("Results of GridSearch")
plt.xlabel("parameters")
plt.ylabel("balanced accuracy of test")
plt.legend()
plt.show()


ind = np.argmax(balanced_test_accuracy)

print(
    "best parameters are : decimation_factor = ",
    parameters[ind][0],
    "epoch_duration = ",
    parameters[ind][1],
    "filter = (",
    parameters[ind][2],
    ",",
    parameters[ind][3],
    ",",
    parameters[ind][4],
    ")",
)

print("parameters: ", parameters)
print("balanced_test_accuracy: ", balanced_test_accuracy)
