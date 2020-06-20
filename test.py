import utils
import numpy as np

patient = utils.PatientLoader(3)



audio = patient.get_audio()
channel_names = patient.get_channels()
seeg = patient.get_seeg()
word = patient.get_words()


kinetics_1 = np.load('/home/eduardo/master/mrp2/kinetics/kh1_1_metrics.npy')

print(patient.get_audio())

