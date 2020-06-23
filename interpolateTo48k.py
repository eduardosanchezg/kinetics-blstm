#!/usr/bin/env python
import os
import sys
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

fname = "kh1_1"
DATA_DIR = "../Data/"

metrics = np.load('data_metrics/'+fname+'_metrics.npy')
EEG = np.load(DATA_DIR + fname + "_sEEG.npy")
audio = np.load(DATA_DIR + fname + "_audio.npy")

calc = audio.shape[0] / EEG.shape[0]

print("EEG shape: ", EEG.shape[0], " audio: ", audio.shape[0], " Multi: ", calc)

# LA = metrics[0,:]
# LP = metrics[1,:]
# TBCL = metrics[2,:]
# TBCD = metrics[3,:]
# TTCL = metrics[4,:]
# TTCD = metrics[5,:]

# # print(LA.shape[0])

# x = np.linspace(0, LA.shape[0]*20, LA.shape[0])
# yLA = LA.T
# yLP = LP.T
# yTBCL = TBCL.T
# yTBCD = TBCD.T
# yTTCL = TTCL.T
# yTTCD = TTCD.T


# xnew = np.linspace(0, LA.shape[0]*20, LA.shape[0]*20)
# fLA = interp1d(x, yLA,kind = 'linear')
# fLP = interp1d(x, yLP,kind = 'linear')
# fTBCL = interp1d(x, yTBCL,kind = 'linear')
# fTBCD = interp1d(x, yTBCD,kind = 'linear')
# fTTCL = interp1d(x, yTTCL,kind = 'linear')
# fTTCD = interp1d(x, yTTCD,kind = 'linear')

# LA1k = fLA(xnew)
# LP1k = fLP(xnew)
# TBCL1k = fTBCL(xnew)
# TBCD1k = fTBCD(xnew)
# TTCL1k = fTTCL(xnew)
# TTCDL1k = fTTCD(xnew)

# output = np.array([LA1k,LP1k,TBCL1k,TBCD1k,TTCL1k,TTCDL1k])

# if not os.path.exists('data_metrics1k'):
#     os.makedirs('data_metrics1k')


# print("processed:" + fname + " shape: " + str(output.shape))

# np.save('data_metrics1k/' + fname+'_metrics.npy', output)

# print(LA48k.shape)
# plt.plot(x[0:100000],yLA[0:100000],'o',xnew[0:100000], LA1k[0:100000], '-')
# plt.show()