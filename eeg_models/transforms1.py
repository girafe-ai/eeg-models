import numpy as np
from scipy import signal
from sklearn.base import BaseEstimator, TransformerMixin

# from eeg_models.types import List, Optional, npArray
from somepytools.typing import (  # npArray  =>  Array   in somepytools.typing
    Array,
    List,
    Optional,
)


class Transform(BaseEstimator, TransformerMixin):
    """Base class for transformers

    Providing dummy implementation of the methods expected by sklearn
    """

    def fit(self, batch: Array, labels: Optional[Array] = None):
        return self


class ButterFilter(Transform):
    """Applies Scipy's Butterworth filter"""

    def __init__(self, sampling_rate: int, order: int, highpass: int, lowpass: int):
        self.sampling_rate = sampling_rate
        self.order = order
        self.highpass = highpass
        self.lowpass = lowpass

        normal_cutoff = tuple(
            a / (0.5 * self.sampling_rate) for a in (self.highpass, self.lowpass)
        )
        self.filter = signal.butter(self.order, normal_cutoff, btype="bandpass")

    def transform(self, batch: Array) -> List[Array]:
        out = np.empty_like(batch)
        out[:] = [signal.filtfilt(*self.filter, item) for item in batch]
        return out


class Decimator(Transform):
    """Performs signal decimation with `scipy.signal.decimate`"""

    def __init__(self, factor: int):
        """
        Args:
            factor: downsampling factor, shouldn't be more than 13,
                see :py:funct:`scipy.signal.decimate` for more info
        """
        self.factor = factor

    def transform(self, batch: Array) -> List[Array]:
        """
        Args:
            batch: iterable of np.ndarrays

        Returns:
            np.ndarray of np.objects shaped (len(x), )
                In other words it outputs ndarray of objects each of which is
                result of decimation of items from x
        """
        out = np.empty(len(batch), dtype=np.object)
        out[:] = [signal.decimate(item, self.factor) for item in batch]
        return out


class ChannellwiseScaler(Transform):
    """Performs channelwise scaling according to given scaler"""

    def __init__(self, scaler: Transform):
        """
        Args:
            scaler: instance of one of sklearn.preprocessing classes
                StandardScaler or MinMaxScaler or analogue
        """
        self.scaler = scaler

    def fit(self, batch: Array, labels: Optional[Array] = None):
        """
        Args:
            batch: array of eegs, that is every element of x is (n_channels, n_ticks)
                batch shaped (n_eegs) of 2d array or (n_eegs, n_channels, n_ticks)
        """
        for signals in batch:
            # self.scaler.partial_fit(signals.T)
            self.scaler.partial_fit(signals.reshape(-1, 1))

        # self.scaler.partial_fit(batch)

        return self

    def transform(self, batch: Array) -> List[Array]:
        """Scales each channel

        Args:
            batch: Data to process, could be either
                * one record, 2-dim input, (n_channels, n_samples)
                * or many records 3-dim, (n_records, n_channels, n_samples)

        Returns:
            the same format as input
        """
        scaled = np.empty_like(batch)
        for i, signals in enumerate(batch):
            # double T for scaling each channel separately
            # scaled[i] = self.scaler.transform(signals.T).T
            scaled[i] = self.scaler.transform(signals.reshape(-1, 1)).squeeze(1)
        return scaled


class MarkersTransformer(Transform):
    """Transforms markers channels to arrays of indexes of epoch start and labels"""

    def __init__(
        self, labels_mapping: dict, decimation_factor: int = 1, empty_label: float = 0.0
    ):
        """
        Args:
            labels_mapping: dict which maps
        """
        self.labels_mapping = labels_mapping
        self.decimation_factor = decimation_factor
        self.empty_label = empty_label

    def transform(self, batch: Array) -> List[Array]:
        res = []

        for markers in batch:
            index_label = []
            for index, label in enumerate(markers):
                if label == self.empty_label:
                    continue

                new_index = index // self.decimation_factor
                new_label = self.labels_mapping[label]
                index_label.append((new_index, new_label))

            res.append(np.array(index_label, dtype=np.int))

        return res
