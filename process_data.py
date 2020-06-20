import numpy as np 
from mne.filter import filter_data
import scipy.io.wavfile as wav
import scipy
from scipy import fftpack
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.stats import mode
import MelFilterBank as mel
from scipy.signal import decimate, hilbert

hilbert3 = lambda x: hilbert(x, fftpack.next_fast_len(len(x)),axis=0)[:len(x)]

def extractHG(data, sr,windowLength=0.05,frameshift=0.01):
    #Linear detrend
    data = scipy.signal.detrend(data,axis=0)
    numWindows=int(np.floor((data.shape[0]-windowLength*sr)/(frameshift*sr)))
    data = filter_data(data.T, sr, 70,170,method='iir').T
    data = filter_data(data.T, sr, 102, 98,method='iir').T # Band-stop
    data = filter_data(data.T, sr, 152, 148,method='iir').T
    data = np.abs(hilbert3(data))
    ecogFeat = np.zeros((numWindows,data.shape[1]))
    for win in range(numWindows):
        startEcog= int(np.floor((win*frameshift)*sr))
        stopEcog = int(np.floor(startEcog+windowLength*sr))
        ecogFeat[win,:] = np.mean(data[startEcog:stopEcog,:],axis=0)
        #for c in range(data.shape[1]):
            #ecogFeat[win,c]=np.log(np.sum(data[startEcog:stopEcog,c]**2)+0.01)
        #    ecogFeat[win,c]=np.mean(np.abs(scipy.signal.hilbert(data[startEcog:stopEcog,c])))
    return ecogFeat

def stackFeatures(features, modelOrder=4, stepSize=5):
    ecogFeatStacked=np.zeros((features.shape[0]-(2*modelOrder*stepSize),(2*modelOrder+1)*features.shape[1]))
    for fNum,i in enumerate(range(modelOrder*stepSize,features.shape[0]-modelOrder*stepSize)):
        ef=features[i-modelOrder*stepSize:i+modelOrder*stepSize+1:stepSize,:]
        ecogFeatStacked[fNum,:]=ef.flatten() # Add 'F' if stacked the same as matlab
    return ecogFeatStacked

def downsampleLabels(labels, sr, windowLength=0.05, frameshift=0.01):
    numWindows=int(np.floor((labels.shape[0]-windowLength*sr)/(frameshift*sr)))
    newLabels = np.empty(numWindows, dtype="S15")
    for w in range(numWindows):
        start = int(np.floor((w*frameshift)*sr))
        stop = int(np.floor(start+windowLength*sr))
        newLabels[w]=mode(labels[start:stop])[0][0].encode("ascii", errors="ignore").decode()
    return newLabels

def windowAudio(audio, sr, windowLength=0.05, frameshift=0.01):
    numWindows=int(np.floor((audio.shape[0]-windowLength*sr)/(frameshift*sr)))
    winAudio = np.zeros((numWindows, int(windowLength*sr )))
    for w in range(numWindows):
        startAudio = int(np.floor((w*frameshift)*sr))
        stopAudio = int(np.floor(startAudio+windowLength*sr))    
        winAudio[w,:] = audio[startAudio:stopAudio]
    return winAudio

def extractMelSpecs(audio, sr, windowLength=0.05, frameshift=0.01):
    numWindows=int(np.floor((audio.shape[0]-windowLength*sr)/(frameshift*sr)))
    win = scipy.hanning(np.floor(windowLength*sr + 1))[:-1]
    spectrogram = np.zeros((numWindows, int(np.floor(windowLength*sr / 2 + 1))),dtype='complex')
    for w in range(numWindows):
        startAudio = int(np.floor((w*frameshift)*sr))
        stopAudio = int(np.floor(startAudio+windowLength*sr))
        a = audio[startAudio:stopAudio]
        spec = np.fft.rfft(win*a)
        spectrogram[w,:] = spec
    mfb = mel.MelFilterBank(spectrogram.shape[1], 23, sr)
    spectrogram = np.abs(spectrogram)
    spectrogram = (mfb.toLogMels(spectrogram)).astype('float')
    return spectrogram

def nameVector(elecs,modelOrder=4):
    names = np.matlib.repmat(elecs.astype(np.dtype(('U', 10))),1,2 * modelOrder +1).T
    for i, off in enumerate(range(-modelOrder,modelOrder+1)):
        names[i,:] = [e[0] + 'T' + str(off) for e in elecs]
    return names.flatten()  # Add 'F' if stacked the same as matlab


if __name__=="__main__":
    winL = 0.05 # 0.01
    frameshift = 0.01 #0.01
    modelOrder=4
    stepSize=5
    path = r'/home/eduardo/master/mrp2/data/'
    outPath = r'/home/eduardo/master/mrp2/out'
    pts = ['kh1', 'kh2', 'kh3','kh4','kh5','kh6','kh7','kh8','kh9']
    pts=['kh9']
    sessions = [1,2,1,1,1,1,1,1,1]
    for pNr, p in enumerate(pts):
        for ses in range(1,sessions[pNr]+1):
            dat = np.load(path + '/' + p + '_' + str(ses) + '_sEEG.npy')
            sr=1024
            if p=='kh9':
                sr=2048
            # Extract HG features
            feat = extractHG(dat,sr, windowLength=winL,frameshift=frameshift)

            # Stack features
            feat = stackFeatures(feat,modelOrder=modelOrder,stepSize=stepSize)
            
            
            # Extract labels
            words=np.load(path + '/' + p + '_' + str(ses) + '_words.npy')
            words=downsampleLabels(words,sr,windowLength=winL,frameshift=frameshift)
            words=words[modelOrder*stepSize:words.shape[0]-modelOrder*stepSize]

            # Load audio
            audio = np.load(path + '/' + p + '_' + str(ses) + '_noisyAudio.npy')
            audioSamplingRate = 48000
            targetSR = 16000
            audio = decimate(audio,int(audioSamplingRate / targetSR))
            audioSamplingRate = targetSR
            scaled = np.int16(audio/np.max(np.abs(audio)) * 32767)
            wav.write(outPath + p + '_' + str(ses) + '_orig_audio.wav',audioSamplingRate,scaled)   

            melSpec = extractMelSpecs(scaled,audioSamplingRate,windowLength=winL,frameshift=frameshift)
            winAudio = windowAudio(scaled, audioSamplingRate,windowLength=winL,frameshift=frameshift)
            # Align to ECoG features
            melSpec = melSpec[modelOrder*stepSize:melSpec.shape[0]-modelOrder*stepSize,:]
            winAudio = winAudio[modelOrder*stepSize:winAudio.shape[0]-modelOrder*stepSize,:]
            if melSpec.shape[0]!=feat.shape[0]:
                print('Possible Problem with ECoG/Audio alignment for %s session %d.' % (p,ses))
                melSpec = melSpec[:feat.shape[0],:]
                winAudio = winAudio[:feat.shape[0],:]
            # Save everything
            np.save(outPath + p + '_' + str(ses) + '_feat.npy', feat)
            np.save(outPath + p + '_' + str(ses) + '_procWords.npy', words)
            np.save(outPath + p + '_' + str(ses) + '_spec.npy',melSpec)
            np.save(outPath + p + '_' + str(ses) + '_winAudio.npy',winAudio)
            
            elecs = np.load(path + p + '_' + str(ses) + '_channelNames.npy')
            np.save(outPath + p + '_' + str(ses) + '_feat_names.npy', nameVector(elecs, modelOrder=modelOrder))