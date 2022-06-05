from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


plt.style.use("seaborn-whitegrid")
sns.set()


def data_to_plot(dataset, subject_id: int, number_sample: int = 1000) -> Dict[str, Any]:
    """
    subroutine to prepare data to plot : for subject subject_id, using number_sample measures from original dataset
    """
    record = dataset._read_hdf(dataset.root / dataset.meta.loc[subject_id, "filename"])
    runs_raw = {}
    for i, act in enumerate(record):
        # target and stims are increased by 1
        # because the channel is filled with zeros by default
        target = act["target"] + 1
        run_data = []
        for eeg, starts, stims in act["sessions"]:
            starts = starts * dataset.sampling_rate / dataset._ms_in_sec
            starts = starts.round().astype(np.int)
            stims = stims + 1
            stims_channel = np.zeros(eeg.shape[1])
            target_channel = np.zeros(eeg.shape[1])

            for start, stimul in zip(starts, stims):
                stims_channel[start] = stimul
                target_channel[start] = 1 if stimul == target else 2

            round_data = np.vstack((eeg, stims_channel[None, :], target_channel[None, :]))
            run_data.append(round_data)

        raw = np.hstack(run_data)
        runs_raw[f"run_{i}"] = raw
    data = {"session_0": runs_raw}

    #  NROWS : number of output epochs of number_sample points starting -100 before point of interest
    NROWS = 170

    data_tab, label_tab, index_tab = _get_data_tab(data, number_sample, nrows=NROWS)

    # preprocess data  :
    filtered_eeg = []
    filtered_label = []
    filtered_index = []
    for i in range(NROWS):
        raw_eegs = data_tab[i]
        dataset.transform.fit(raw_eegs)
    for i in range(NROWS):
        raw_eegs = data_tab[i]
        result = np.stack(dataset.transform.transform(raw_eegs))
        filtered_eeg.append(result)
        filtered_label.append(label_tab[i])
        filtered_index.append(index_tab[i])
    output = (np.stack(filtered_eeg), np.stack(filtered_label), np.stack(filtered_index))
    return output


def _get_data_tab(data, number_sample, nrows):

    # number_sample  :   number of samples before decimation
    # nrows : number of sub-epochs extrated from each subject file

    data_tab = []
    label_tab = []
    index_tab = []
    NROWS = nrows

    # for each session in subject file
    for i in range(len(data)):
        #  for each run of a session
        for j in range(len(data[f"session_{i}"])):
            data_tmp = []
            label_tmp = []
            index_tmp = []
            i_low = 200
            i_high = i_low + number_sample
            m_low = -100
            m_high = m_low + number_sample
            # find event of interest
            signal = data[f"session_{i}"][f"run_{j}"][9]
            signal_test = [
                (indice, signal[indice])
                for indice in range(len(signal))
                if signal[indice] == 1
            ]
            # for each event create one epoch of self.sample_per_batch samples
            if signal_test == []:
                for k in range(10):
                    # limits = ( 300 + k * 1400, 700 + k * 1400)
                    limits = (i_low + k * 1400, i_high + k * 1400)
                    if (
                        data[f"session_{i}"][f"run_{j}"][
                            0:8, limits[0] : limits[1]
                        ].shape[1]
                        == number_sample
                    ):
                        data_tmp.append(
                            data[f"session_{i}"][f"run_{j}"][0:8, limits[0] : limits[1]]
                        )
                        label_tmp.append(np.array([0.0]))
                        index_tmp.append(np.array([limits[0], limits[1]]))
            if signal_test != []:
                for k in range(len(signal_test)):
                    current_indice = signal_test[k][0]
                    if (
                        data[f"session_{i}"][f"run_{j}"][
                            0:8, current_indice + m_low : current_indice + m_high
                        ].shape[1]
                        == number_sample
                    ):
                        data_tmp.append(
                            data[f"session_{i}"][f"run_{j}"][
                                0:8, current_indice + m_low : current_indice + m_high
                            ]
                        )
                        label_tmp.append(np.array([1.0]))
                        index_tmp.append(
                            np.array([current_indice + m_low, current_indice + m_high])
                        )
            data_tab += data_tmp
            label_tab += label_tmp
            index_tab += index_tmp
        data_tab = (np.stack(data_tab))[:NROWS]
        label_tab = (np.stack(label_tab))[:NROWS]
        index_tab = (np.stack(index_tab))[:NROWS]
    return data_tab, label_tab, index_tab


def multiple_plot_data(dataset, subject_id: int, number_sample: int, index: List):
    """
    plot signal for inputs :
        - dataset :  dataset to vizualize
        - subject_id : index of the subject
        - number_sample :  number of continous samples in original dataset
        - index : list of indices of epochs

    output : plot of
        - 8 channel signals of one epoch in the list : number_sample points displayed
        - label value for epoch in list
        - indexes values of epoch window : starting sample indice, event indice, ending sample indice
    """
    data, label, indexes = data_to_plot(dataset, subject_id, number_sample)
    figure, axis = plt.subplots(len(index) * 2 + 1, 1)
    for i in range(len(index)):
        x_plot = np.linspace(0, data[index[i]].shape[1], data[index[i]].shape[1])
        y_plot = data[index[i]].T
        axis[i * 2].plot(x_plot, y_plot)
        axis[i * 2].set_title(
            f"epoch : {index[i]} - label : {label[index[i]]} - indices : {indexes[index[i]]}"
        )
        y_plot = np.mean(y_plot, axis=1)
        axis[i * 2 + 1].plot(x_plot, y_plot)
        axis[i * 2 + 1].set_title(
            f"mean epoch : {index[i]} - label : {label[index[i]]} - indices : {indexes[index[i]]}"
        )
    y_plot = np.zeros_like(data[index[0]])
    for i in range(len(index)):
        y_plot += data[index[i]]
    y_plot = y_plot.T / len(index)
    y_plot = np.mean(y_plot, axis=1)
    axis[len(index) * 2].plot(x_plot, y_plot)
    axis[len(index) * 2].set_title(" mean over all ploted epochs and over all channels  ")
    plt.show()
    return
