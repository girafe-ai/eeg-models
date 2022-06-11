import numpy as np
from matplotlib import pyplot as plt

from eeg_models.train_braininvaders import EegTraining


sampling_rate = 512
batch_size = 20
validation_split = 0.2
n_epochs = 20


def search_parameters(
    decimation_factor=10, epoch_duration=0.9, order=4, highpass=0.5, lowpass=20
):

    interval_decimation_factor = list(range(6, 13))
    interval_epoch_duration = np.linspace(0.7, 1.3, 7)
    interval_order = list(range(2, 6))
    interval_highpass = np.linspace(0.2, 0.9, 8)
    interval_lowpass = list(range(15, 23))

    f1_score_decimation_factor = []
    f1_score_epoch_duration = []
    f1_score_order = []
    f1_score_highpass = []
    f1_score_lowpass = []
    # decimation factor
    for i in interval_decimation_factor:
        decimation_factor = i
        filter = (order, highpass, lowpass)
        print(
            "decimation_factor = ",
            decimation_factor,
            "epoch_duration = ",
            epoch_duration,
            "filter = (",
            order,
            ",",
            highpass,
            ",",
            lowpass,
            ")",
        )
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
        f1_score_decimation_factor.append(train_braininvaders.results)
    plt.figure(figsize=(15, 15))
    plt.plot(
        interval_decimation_factor,
        f1_score_decimation_factor,
        marker=".",
        label="f1_score_decimation_factor",
    )
    plt.yticks(np.arange(0.2, 0.8, 0.2))
    plt.title("GridSearch for decimation factor")
    plt.xlabel("decimation factors")
    plt.ylabel("f1 score of test")
    plt.legend()
    plt.show()
    ind = np.argmax(f1_score_decimation_factor)
    decimation_factor = interval_decimation_factor[ind]
    print("best decimation factor = ", decimation_factor)
    # epoch_duration
    for i in interval_epoch_duration:
        epoch_duration = i
        filter = (order, highpass, lowpass)
        print(
            "decimation_factor = ",
            decimation_factor,
            "epoch_duration = ",
            epoch_duration,
            "filter = (",
            order,
            ",",
            highpass,
            ",",
            lowpass,
            ")",
        )
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
        f1_score_epoch_duration.append(train_braininvaders.results)
    plt.figure(figsize=(15, 15))
    plt.plot(
        interval_epoch_duration,
        f1_score_epoch_duration,
        marker=".",
        label="f1_score_epoch_duration",
    )
    plt.yticks(np.arange(0.2, 0.8, 0.2))
    plt.title("GridSearch for epoch duration")
    plt.xlabel("epoch duration")
    plt.ylabel("f1 score of test")
    plt.legend()
    plt.show()
    ind = np.argmax(f1_score_epoch_duration)
    epoch_duration = interval_epoch_duration[ind]
    print("best epoch duration = ", epoch_duration)
    # order
    for i in interval_order:
        order = i
        filter = (order, highpass, lowpass)
        print(
            "decimation_factor = ",
            decimation_factor,
            "epoch_duration = ",
            epoch_duration,
            "filter = (",
            order,
            ",",
            highpass,
            ",",
            lowpass,
            ")",
        )
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
        f1_score_order.append(train_braininvaders.results)
    plt.figure(figsize=(15, 15))
    plt.plot(interval_order, f1_score_order, marker=".", label="f1_score_order")
    plt.yticks(np.arange(0.2, 0.8, 0.2))
    plt.title("GridSearch for order")
    plt.xlabel("order")
    plt.ylabel("f1 score of test")
    plt.legend()
    plt.show()
    ind = np.argmax(f1_score_order)
    order = interval_order[ind]
    print("best order = ", order)
    # highpass
    for i in interval_highpass:
        highpass = i
        filter = (order, highpass, lowpass)
        print(
            "decimation_factor = ",
            decimation_factor,
            "epoch_duration = ",
            epoch_duration,
            "filter = (",
            order,
            ",",
            highpass,
            ",",
            lowpass,
            ")",
        )
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
        f1_score_highpass.append(train_braininvaders.results)
    plt.figure(figsize=(15, 15))
    plt.plot(interval_highpass, f1_score_highpass, marker=".", label="f1_score_highpass")
    plt.yticks(np.arange(0.2, 0.8, 0.2))
    plt.title("GridSearch for highpass")
    plt.xlabel("highpass")
    plt.ylabel("f1 score of test")
    plt.legend()
    plt.show()
    ind = np.argmax(f1_score_highpass)
    highpass = interval_highpass[ind]
    print("best highpass = ", highpass)
    # lowpass
    for i in interval_lowpass:
        lowpass = i
        filter = (order, highpass, lowpass)
        print(
            "decimation_factor = ",
            decimation_factor,
            "epoch_duration = ",
            epoch_duration,
            "filter = (",
            order,
            ",",
            highpass,
            ",",
            lowpass,
            ")",
        )
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
        f1_score_lowpass.append(train_braininvaders.results)
    plt.figure(figsize=(15, 15))
    plt.plot(interval_lowpass, f1_score_lowpass, marker=".", label="f1_score_lowpass")
    plt.yticks(np.arange(0.2, 0.8, 0.2))
    plt.title("GridSearch for lowpass")
    plt.xlabel("lowpass")
    plt.ylabel("f1 score of test")
    plt.legend()
    plt.show()
    ind = np.argmax(f1_score_lowpass)
    lowpass = interval_lowpass[ind]
    print("best lowpass = ", lowpass)
    print(
        "best parameters are : decimation_factor = ",
        decimation_factor,
        "epoch_duration = ",
        epoch_duration,
        "filter = (",
        order,
        ",",
        highpass,
        ",",
        lowpass,
        ")",
    )


search_parameters()
