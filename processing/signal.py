import numpy as np

from scipy import fftpack
from scipy.signal import decimate
from scipy.signal import detrend
from scipy.signal import hilbert
from mne.filter import filter_data


def rescale(data, low=-1, high=1):
    """ Credit towards https://en.wikipedia.org/wiki/Feature_scaling """
    min_val = data.min()
    shifted = data - min_val
    val_range = data.max() - min_val
    return low + (shifted * (high - low) / val_range)

def hilbert3(x): return hilbert(
    x, fftpack.next_fast_len(len(x)), axis=0)[:len(x)]


def analytic_amp(x): return np.abs(hilbert3(x))


def high_gamma(data, fs, **kwargs):
    data = detrend(data, axis=0).T
    data = filter_data(data, fs,
                       l_freq=70,
                       h_freq=170,
                       method='iir',
                       verbose='WARNING',
                       **kwargs)
    data = filter_data(data, fs,
                       l_freq=102,
                       h_freq=98,
                       method='iir',
                       verbose='WARNING',
                       **kwargs)
    data = filter_data(data, fs,
                       l_freq=152,
                       h_freq=148,
                       method='iir',
                       verbose='WARNING',
                       **kwargs)
    return data.T


def low_component(data, fs,
                  cutoff=30,
                  filt_order=5,
                  **kwargs):
    fparams = dict(ftype='butter', order=filt_order)
    data = detrend(data, axis=0).T
    data = filter_data(data, fs,
                       l_freq=None,
                       h_freq=cutoff,
                       method='iir',
                       iir_params=fparams,
                       verbose='WARNING',
                       **kwargs)
    return data.T
