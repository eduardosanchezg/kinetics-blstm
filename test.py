import utils

patient = utils.PatientLoader(1)

audio = patient.get_audio()
channel_names = patient.get_channels()
seeg = patient.get_seeg()
word = patient.get_words()

print(patient.get_audio())