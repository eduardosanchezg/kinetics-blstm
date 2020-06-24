from argparse import ArgumentParser
from pathlib import Path

import constants as const
import numpy as np
import librosa

from utils import iter_sessions
from processing.speech import extract_world_parameters
from processing.signal import rescale

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--frame-period", type=int)
    parser.add_argument("--target-fs", type=int)
    parser.add_argument("--baldey-dataset", action="store_true")
    return parser.parse_args()

def generate_world(frame_period, target_fs):
    for session in iter_sessions(log=True):
        audio = session.get_audio()
        params = extract_world_parameters(audio,
                                          target_fs=target_fs,
                                          frame_period=frame_period)
        name = f'{const.WORLD_PARAM_NAME}_{frame_period}'
        name = session.get_file_format().format(name)
        np.save(name, params)
    
def generate_world_baldey(frame_period, target_fs):
    baldey_wav_files = Path(const.BALDEY_AUDIO_DIR).rglob("*.wav")
    for file in baldey_wav_files:
        print(f'Processing file {file}')
        audio, fs = librosa.load(file, sr=None)
        # Magic numbers to make it of similar range with our dataset
        rescaled = rescale(audio, low=-0.7, high=0.7)
        params = extract_world_parameters(rescaled,
                                          fs=fs,
                                          target_fs=target_fs,
                                          frame_period=frame_period)
        name = f'{file.parent.stem}_{file.stem}'
        name = f'{name}_world_{frame_period}.npy'
        name = Path(const.BALDEY_KINETICS_DIR) / name
        print(f'Writing file {name}', end='\n\n')
        np.save(name, params)
    
if __name__ == "__main__":
    args = parse_args()
    if args.baldey_dataset:
        generate_world_baldey(args.frame_period, args.target_fs)
    else:
        generate_world(args.frame_period, args.target_fs)
    