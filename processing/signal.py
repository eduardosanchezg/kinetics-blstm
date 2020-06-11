import numpy as np

from scipy import fftpack
from scipy.signal import decimate
from scipy.signal import detrend
from scipy.signal import hilbert
from mne.filter import filter_data


def hilbert3(x): return hilbert(
    x, fftpack.next_fast_len(len(x)), axis=0)[:len(x)]


def analytic_amp(x): return np.abs(hilbert3(x))


def high_gamma(data, fs, **kwargs):
    data = detrend(data, axis=0).T
    data = filter_data(data, fs,
                       l_freq=70,
                       h_freq=170,
                       method='iir',
                       **kwargs)
    data = filter_data(data, fs,
                       l_freq=102,
                       h_freq=98,
                       method='iir',
                       **kwargs)
    data = filter_data(data, fs,
                       l_freq=152,
                       h_freq=148,
                       method='iir',
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
                       **kwargs)
    return data.T
