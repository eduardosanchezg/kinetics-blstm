import numpy as np
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import MelFilterBank as mel
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import reconstructWave as rW
import scipy.io.wavfile as wavefile


path = r'C:/data/processed/LSL-Speech/features/'
pts = ['kh1', 'kh2', 'kh3','kh4','kh5','kh6','kh7','kh8','kh9']
#pts=['kh9']
sessions = [1,2,1,1,1,1,1,1]
for pNr, pt in enumerate(pts):
    for ses in range(1,sessions[pNr]+1):

        spectrogram = np.load(path + pt + '_' + str(ses) + '_spec.npy')
        data = np.load(path + pt + '_' + str(ses) + '_feat.npy')
        labels = np.load(path + pt + '_' + str(ses) + '_procWords.npy')
        featName = np.load(path + pt + '_' + str(ses) + '_feat_names.npy')
        nfolds =10
        winLength=0.05
        frameshift=0.01 #0.025
        audiosr=16000

        rec_spec = np.zeros(spectrogram.shape)
        coefs=np.zeros((data.shape[1],spectrogram.shape[1],nfolds))
        kf = KFold(nfolds,shuffle=False)
        #from unitSelectionRegressor import UnitSelectionRegressor
        #est=UnitSelectionRegressor(context=0,alpha=0)
        #from gmmMapping import GMMRegression
        #est=GMMRegression(k=4)
        est=LinearRegression(n_jobs=5)
        pca=PCA()
        for k,(train, test) in enumerate(kf.split(data)):
            #Z-Normalize
            mu=np.mean(data[train,:],axis=0)
            std=np.std(data[train,:],axis=0)
            trainData=(data[train,:]-mu)/std
            testData=(data[test,:]-mu)/std
            #pca.fit(trainData)
            #numComps=np.argwhere(np.cumsum(pca.explained_variance_ratio_)>.5)[0][0]
            #print(numComps)

            #trainData=np.dot(trainData, pca.components_[:numComps,:].T)
            #testData = np.dot(testData, pca.components_[:numComps,:].T)
            #Select channels
            cs=np.zeros(data.shape[1])
            for f in range(data.shape[1]):
                if not np.any(np.isnan(trainData[:,f])):
                    cs[f],p=spearmanr(trainData[:,f],np.mean(spectrogram[train,:],axis=1))
            select=np.argsort(np.abs(cs))[np.max([-40,-len(cs)]):]
            #select=np.arange(trainData.shape[1])
            #print(elecNames[select])
            #print(cs[np.argsort(cs)[-200:]])
            est.fit(trainData[:, select], spectrogram[train, :])
            #est.fit(data[train,:][:,select],spectrogram[train,:],max_iters=1000)
            rec_spec[test, :] = est.predict(testData[:, select])

        rs=[]
        rsVAD=[]
        if np.any(np.isnan(rec_spec)):
            print('%s session %d has %d broken samples in recontruction' % (pt, ses, np.sum(np.isnan(rec_spec))))
        rec_spec[np.isnan(rec_spec)]=0
        for specBin in range(spectrogram.shape[1]):
            #r,p=spearmanr(spectrogram[:,specBin],rec_spec[:,specBin])
            r, p = pearsonr(spectrogram[:, specBin], rec_spec[:, specBin])
            rs.append(r)

        print('%s session %d has mean correlation of %f' % (pt, ses, np.mean(rs)))
        #plt.plot(rs)
        #plt.show()


        cm='viridis'
        fig, ax = plt.subplots(2, sharex=True)
        pSta=int(1*(1/frameshift));pSto=int(30*(1/frameshift))
        ax[0].imshow(np.flipud(spectrogram[pSta:pSto, :].T), cmap=cm, interpolation=None,aspect='auto')
        ax[0].set_ylabel('Log Mel-Spec Bin')
        ax[1].imshow(np.flipud(rec_spec[pSta:pSto, :].T), cmap=cm, interpolation=None,aspect='auto')
        plt.setp(ax[1], xticks=np.arange(0,pSto-pSta,int(1/frameshift)), xticklabels=[str(x/int(1/frameshift)) for x in np.arange(0,pSto-pSta,int(1/frameshift))])
        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('Log Mel-Spec Bin')
        #plt.show()


        mfb = mel.MelFilterBank(401, 23, audiosr)
        wavPath = r'C:/data/processed/LSL-Speech/results/'
        hop=int(spectrogram.shape[0]/nfolds)
        rec_audio=np.array([])

        for_reconstruction=mfb.fromLogMels(rec_spec)

                    #for_reconstruction[:,:8]=0
        for w in range(0,spectrogram.shape[0],hop):
            #spec=np.exp(spectrogram[w:min(w+hop,spectrogram.shape[0])])
            spec=for_reconstruction[w:min(w+hop,for_reconstruction.shape[0]),:]
            #if spec.shape[0]%2==1:
            #    spec=spec[:-1,:]
            rec=rW.reconstructWavFromSpectrogram(spec,spec.shape[0]*spec.shape[1],fftsize=int(audiosr*winLength),overlap=int(winLength/frameshift))
            rec_audio=np.append(rec_audio,rec)
        scaled = np.int16(rec_audio/np.max(np.abs(rec_audio)) * 32767)
        scaled = np.int16(rec_audio)
        wavefile.write(wavPath + 'test' + str(pt) + '_' + str(ses) + '.wav',int(audiosr),scaled)
        #for_reconstruction=mfb.fromLogMels(spectrogram)
        #origWav=np.array([])
        #for w in range(0,for_reconstruction.shape[0],hop):
        #    #spec=np.exp(spectrogram[w:min(w+hop,spectrogram.shape[0])])
        #    spec=for_reconstruction[w:min(w+hop,for_reconstruction.shape[0])]
        #    rec=rW.reconstructWavFromSpectrogram(spec,spec.shape[0]*spec.shape[1],fftsize=int(audiosr*winLength),overlap=int(winLength/frameshift))
        #    origWav=np.append(origWav,rec)
        #origWav = np.int16(origWav / np.max(np.abs(origWav)) * 32767)
        #wavefile.write(wavPath + 'orig'+ str(sub) +'.wav',int(audiosr),origWav)

