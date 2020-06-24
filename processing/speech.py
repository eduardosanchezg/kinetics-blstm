import numpy as np
import pyworld as pw

from numpy import ascontiguousarray as ctg_array
from scipy.io.wavfile import write as write_wav
from librosa import resample

from constants import AUDIO_fs


def extract_world_parameters(audio,
                             fs=AUDIO_fs,
                             target_fs=16000,
                             frame_period=10, # ms
                             return_individual=False):
    downsampled = resample(audio, orig_sr=fs, target_sr=target_fs)
    downsampled = ctg_array(downsampled, dtype=np.double)

    f0, t = pw.harvest(x=downsampled,
                       fs=target_fs,
                       frame_period=frame_period)
    sp = pw.cheaptrick(x=downsampled,
                       f0=f0,
                       temporal_positions=t,
                       fs=target_fs)
    ap = pw.d4c(x=downsampled,
                f0=f0,
                temporal_positions=t,
                fs=target_fs)
    
    encoded_ap = pw.code_aperiodicity(aperiodicity=ap,
                                      fs=target_fs)
    f0 = f0.reshape(-1, 1)
    vuv = (f0 > 0).astype(int)

    if return_individual:
        return dict(sp=sp, encoded_ap=encoded_ap, f0=f0, vuv=vuv)
    
    return np.concatenate((sp, encoded_ap, f0, vuv), axis=1)


def split_world_param(param):
    return dict(sp=ctg_array(param[:, :513], dtype=np.double),
                encoded_ap=ctg_array(param[:, [513]], dtype=np.double),
                f0=ctg_array(param[:, [514]], dtype=np.double),
                vuv=ctg_array(param[:, [515]], dtype=np.double))


def world_reconstruct_audio(sp,
                            encoded_ap,
                            f0,
                            vuv,
                            fs=16000,
                            frame_period=10,
                            fft_size=1024):
    f0 = f0.reshape(-1)
    encoded_ap = encoded_ap.reshape(-1, 1)
    encoded_ap = ctg_array(encoded_ap, dtype=np.float64)
    decoded_ap = pw.decode_aperiodicity(encoded_ap, fs, fft_size)

    audio = pw.synthesize(f0=f0,
                          spectrogram=sp,
                          aperiodicity=decoded_ap,
                          fs=fs,
                          frame_period=frame_period)
    scaled_audio = np.int16(audio / np.max(np.abs(audio)) * 32767)
    return scaled_audio
