import numpy as np
import pyworld as pw

from numpy import ascontiguousarray as ctg_array
from scipy.io.wavfile import write as write_wav
from scipy.signal import decimate

from constants import AUDIO_fs

def extract_world_parameters(audio,
                             fs=AUDIO_fs,
                             target_fs=16000,
                             frame_period=10, # ms
                             return_individual=False):
    downsampled = decimate(audio, int(fs / target_fs))
    downsampled = ctg_array(downsampled, dtype=np.float64)

    f0, t = pw.harvest(x=downsampled,
                       fs=target_fs,
                       frame_period=frame_period)
    sp = pw.cheaptrick(x=downsampled,
                       f0=f0,
                       temporal_positions=t,
                       fs=target_fs)
    ap = pw.d4c(x=audio,
                f0=f0,
                temporal_positions=t,
                fs=target_fs)
    
    encoded_ap = pw.code_aperiodicity(aperiodicity=ap,
                                      fs=target_fs)
    f0 = f0.reshape(-1, 1)
    vuv = (f0 > 0).astype(int)

    if return_individual:
        return sp, encoded_ap, f0, vuv
    
    return np.concatenate((sp, encoded_ap, f0, vuv), axis=1)

def world_reconstruct_audio(file_name, fs, frame_period, sp, encoded_ap, f0, vuv, fft_size=1024):
    f0 = f0.reshape(-1)
    ap = ctg_array(encoded_ap, dtype=np.float64)
    ap = ap.reshape(-1, 1)
    decoded_ap = pw.decode_aperiodicity(aperiodicity=ap, 
                                        fs=fs,
                                        fft_size=fft_size)

    audio = pw.synthesize(f0=f0,
                          spectrogram=sp,
                          aperiodicity=decoded_ap,
                          fs=fs,
                          frame_period=frame_period)
    scaled_audio = np.int16(audio / np.max(np.abs(audio)) * 32767)
    write_wav(file_name, fs, scaled_audio)
