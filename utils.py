import os
import numpy as np

from constants import DATA_DIR, SEEG_fs

class PatientLoader:

    _seeg = 'sEEG'
    _ch_names = 'channelNames'
    _words = 'words'
    _audio = 'audio'
    _noisy = 'noisyAudio'

    def __init__(self, patient_id, session=1, data_dir=DATA_DIR):
        if type(patient_id) == int:
            patient_id = f'kh{patient_id}'
        self._patient_id = patient_id
        self._session = session
        self._data_dir = data_dir.rstrip(os.path.sep)
        self._file_placeholder = self._make_placeholder()

    @property
    def seeg_fs(self):
        if self._patient_id == 'kh9':
            return 2 * SEEG_fs
        return SEEG_fs

    def get_seeg(self):
        return self._get(PatientLoader._seeg)

    def get_channels(self):
        return self._get(PatientLoader._ch_names)

    def get_words(self):
        return self._get(PatientLoader._words)

    def get_audio(self):
        return self._get(PatientLoader._audio)

    def get_noisy_audio(self):
        return self._get(PatientLoader._noisy)

    def get_features(self, feature_name):
        return self._get(feature_name)

    def get_file_format(self, extension = '.npy', directory=None):
        """ Useful to create new files for the same patient.
        Usage: `
        """
        file_format = self._file_placeholder
        if extension != '.npy':
            file_format = file_format.replace('.npy', extension)
        if directory is not None:
            directory = directory.rstrip(os.path.sep)
            file_format = file_format.replace(self._data_dir, directory)
        return file_format

    def _get(self, data_file):
        file = self._file_placeholder.format(data_file)
        return np.load(file)
    
    def _make_placeholder(self):
        file_name = f'{self._patient_id}_{self._session}_{{}}.npy'
        return self._data_dir  + os.path.sep + file_name
         