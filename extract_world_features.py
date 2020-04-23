#!/usr/bin/env python

import constants

import ray
import numpy as np

from utils import PatientLoader
from processing.speech import extract_world_parameters, world_reconstruct_audio

if __name__ == "__main__":
    sessions = [(1, 1),
                (2, 1),
                (2, 2),
                (3, 1),
                (4, 1),
                (5, 1),
                (6, 1),
                (7, 1),
                (8, 1),
                (9, 1),
                (10, 1)]

    @ray.remote
    def process_session(patient, session):
        print(f'Extracting WORLD features of session {session} from patient {patient}.')
        loader = PatientLoader(patient_id=patient, session=session)
        features = extract_world_parameters(loader.get_audio()) # defaults
        
        print(f'Saving WORLD features of session {session} from patient {patient}.')
        file_name = loader.get_file_format().format(constants.WORLD_PARAM_NAME)
        np.save(file_name, features)

    ray.init()
    
    ids = [process_session.remote(*s) for s in sessions]

    ray.wait(ids, num_returns=len(ids))

    for patient, session in sessions:
        print(f'Extracting WORLD features of session {session} from patient {patient}.')
        loader = PatientLoader(patient_id=patient, session=session)
        features = extract_world_parameters(loader.get_audio()) # defaults
        
        print(f'Saving WORLD features of session {session} from patient {patient}.')
        file_name = loader.get_file_format().format(constants.WORLD_PARAM_NAME)
        np.save(file_name, features)

    synthesis_sessions = np.random.choice(len(sessions), 2)
    for s in synthesis_sessions:
        patient, session = sessions[s]
        
        print(f'Re-synthesizing {session} from patient {patient} speech from WORLD parameters.')
        loader = PatientLoader(patient_id=patient, session=session)
        features = extract_world_parameters(loader.get_audio(), return_individual=True)
        file_name = loader.get_file_format(extension='.wav').format(constants.WORLD_RECONSTRUCTION_NAME)
        world_reconstruct_audio(file_name, **features)
