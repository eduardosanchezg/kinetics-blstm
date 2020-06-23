#!/usr/bin/env python
import os
import glob
import sys
import numpy as np
import scipy
import scipy.io
import librosa
import subprocess
import HTK
from contextualize import *
from writehtk import *
from subprocess import call
import matplotlib.pyplot as plt
from KalmanSmoother import *
import pickle

DATA_DIR = "../Data/" # MRP Data

Baldey_Dir = "../baldey_audio/real_underived/"

def float2pcm16(f):    
    f = f * 32768 ;
    f[f > 32767] = 32767;
    f[f < -32768] = -32768;
    i = np.int16(f)
    return i

def estimate_tv_xrmb(fname, mrp):
#=========================================================================
# Function to estimate TVs from pretrained TV estimator
# Usage:        estimate_tv_xrmb(fpath)
# Inputs:       fpath - string specifying path to input wav file
# Outputs:      Saved mat file <filename_tv.mat> in the current directory
# Author:       Ganesh Sivaraman
# Date Created: 11/14/2016
# Date Modified:
#=========================================================================
    WAV_SRATE = 48000   # Signals will be downsampled to 8kHz
    std_frac = 0.25
    CONTEXT = 8
    SKIP = 2

    fname = os.path.splitext(fname)[0]

    temp_fpath = "./temp_data/"+fname+"_audio.wav"

    if not os.path.exists('./temp_data'):
        os.makedirs('./temp_data')

    if mrp:
        sig = np.load(DATA_DIR + fname + "_audio.npy") # patient
    else:
        call(["sox", Baldey_Dir + fname +".wav", temp_fpath])
        sig, fs = librosa.load(temp_fpath, sr=WAV_SRATE)
    print(sig.shape)

    # To avoid clipping and normalize the maximum loudness, divide signal by
    # max of absolute amplitude
    sig = sig/max(abs(sig))
    sig = float2pcm16(sig)
    #print("pcm16", sig.shape)
    scipy.io.wavfile.write(temp_fpath, WAV_SRATE, sig)
    
    # Create HTK MFCC features
    cmd = 'HCopy -T 0 -C mfcc13.conf '+temp_fpath+' '+'./temp_data/'+fname+'_audio.htk'
    subprocess.call(cmd.split(' '))
    
    # read HTK MFCC features and perform mvn
    htkfile = HTK.HTKFile()
    htkfile.load('./temp_data/'+fname+'_audio.htk')
    feats = np.asarray(htkfile.data)
    #print("feats", feats.shape)
    #print(feats.shape)
    mean_G = np.mean(feats, axis=0)
    std_G = np.std(feats, axis=0)
    feats = std_frac*(feats-mean_G)/std_G
    feats = feats.T
    #print("feats", feats.shape)
    cont_feats = contextualize(feats,CONTEXT,SKIP)
    #print("cont", cont_feats.shape)
    with open('xrmb_si_dnn_512_512_512_withDrop_bestmodel_weights.pkl', 'rb') as pickle_file:
        W = pickle.load(pickle_file, fix_imports=True, encoding="latin1")
    out = cont_feats.T
    for ii in [0, 2, 4]:
        out = np.tanh(np.dot(out, W[ii])+W[ii+1])
    tv = np.dot(out, W[6]) + W[7]
    tv = tv.T/std_frac

    #print("tv: " + fname + " shape: " + str(tv.shape))
    tv_smth = kalmansmooth(tv)
    tv_smth = tv_smth[range(0,18,3), :]
    print("processed: " + fname + " shape: " + str(tv_smth.shape))

    # # PLOT!
    # fig, axs = plt.subplots(6)
    # #for y in range(tv_smth.shape[0]): # tv_smth.shape[0]
    # axs[0].plot(tv_smth[0,:10000],color='blue')   
    # axs[1].plot(tv_smth[1,:10000],color='blue')  
    # axs[2].plot(tv_smth[2,:10000],color='blue') 
    # axs[3].plot(tv_smth[3,:10000],color='blue')  
    # axs[4].plot(tv_smth[4,:10000],color='blue')  
    # axs[5].plot(tv_smth[5,:10000],color='blue')  
    # #axs[1].plot(tv[0,:],color='red')          
    # plt.show()

    if mrp: 
        if not os.path.exists('data_metrics'):
            os.makedirs('data_metrics')
        np.save('data_metrics/' + fname+'_metrics.npy', tv_smth)
    else: 
        if not os.path.exists('data_metrics_baldey'):
            os.makedirs('data_metrics_baldey')
        np.save('data_metrics_baldey/' + fname+'_metrics.npy', tv_smth)

    # opdir = opdir
    # opfnm = opdir+'/'+fname+'.mat'
    # writehtk(tv_smth.T, 10, opfnm)
    # outvar = {}
    # outvar['TV'] = tv_smth.T
    # scipy.io.savemat(opfnm, outvar)
    return
    
    
if __name__ == "__main__":
    # patients = ['kh1_1', 'kh2_1', 'kh2_2', 'kh3_1', 'kh4_1', 'kh5_1', 'kh6_1', 'kh7_1', 'kh9_1'] 
    # for i in range(len(patients)):
    #     estimate_tv_xrmb(patients[i], True)

    for file_name in glob.iglob(Baldey_Dir + '*.wav'):
        name = os.path.basename(file_name)
        estimate_tv_xrmb(name, False)

