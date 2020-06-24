import numpy as np
from scipy.signal.windows import hanning
from scipy.signal import resample
from numpy.fft import rfft
from scipy.io.wavfile import read as wavread


def stoi(x, y, fs_signal=10000):
    """
    Returns the output of the short-time objective intelligibility (STOI) measure, where x
    and y denote the clean and processed speech, respectively, with sample rate fs_signal
    in Hz. The output d is expected to have a monotonic relation with the subjective
    speech-intelligibility, where a higher d  denotes better intelligible speech.
    """

    def third_oct(fs, N_fft, num_bands, mn):
        """ returns 1/3 octave band matrix """

        f = np.linspace(0, fs, N_fft+1)
        f = f[0:N_fft // 2 +1]
        k = np.arange(0, num_bands)
        cf = 2 ** (k / 3) * mn
        fl = np.sqrt( (2 ** (k / 3) * mn) * (2 ** ((k-1)/3) * mn) )
        fr = np.sqrt( (2 ** (k / 3) * mn) * (2 ** ((k+1)/3) * mn) )

        A = np.zeros((num_bands, len(f)))

        for i in range(len(cf)):
            b = np.argmin(np.power(f-fl[i], 2))
            fl[i] = f[b]
            fl_ii = b

            b = np.argmin(np.power(f-fr[i], 2))
            fr[i] = f[b]
            fr_ii = b
            A[i, fl_ii:fr_ii] = 1

        rnk = np.sum(A, axis=1)

        # find((rnk(2:end)>=rnk(1:(end-1))) & (rnk(2:end)~=0)~=0, 1, 'last' )+1;
        num_bands = np.where((rnk[1:] >= rnk[0:-1]) & (rnk[1:] != 0) != 0)[0][-1] + 2

        A = A[0:num_bands, :]
        cf = cf[0:num_bands]
        return A

    def stdft(x, N, K, N_fft):
        """
        Returns the short-time hanning-windowed dft of X with frame-size N, overlap K and DFT
        size N_FFT. The columns and rows of X_STDFT denote the frame-index and dft-bin index,
        respectively. Implementation uses rfft to omit the negative frequency terms.
        """
        frames = np.arange(0, len(x) - N, K, dtype=np.int64)
        x_stdft = np.zeros((len(frames), N_fft // 2 + 1),dtype=complex)
        w = hanning(N)

        for i in range(len(frames)):
            ii = np.arange(frames[i], frames[i]+N, dtype=np.int64)
            x_stdft[i, :] = rfft(x[ii] * w, N_fft)

        return x_stdft

    def remove_silent_frames(x, y, r, N, K):
        """
        X and Y are segmented with frame-length N and overlap K, where the maximum energy
        of all frames of X is determined, say X_MAX. X_SIL and Y_SIL are the reconstructed
        signals, excluding the frames, where the energy of a frame of X is smaller than
        X_MAX-RANGE
        """

        frames = np.arange(0, len(x)-N, K)
        w = hanning(N)
        msk = np.zeros(len(frames))

        for j in np.arange(0, len(frames)):
            jj = np.arange(frames[j], frames[j]+N, dtype=np.int64)
            msk[j] = 20 * np.log10(np.linalg.norm(x[jj] * w) / np.sqrt(N))

        msk = (msk - max(msk) + r) > 0
        count = 0

        x_sil = np.zeros(len(x))
        y_sil = np.zeros(len(y))

        last_index = len(x)
        for j in range(0, len(frames)):
            if msk[j]:
                jj_i = np.arange(frames[j], frames[j]+N, dtype=np.int64)
                jj_o = np.arange(frames[count], frames[count]+N, dtype=np.int64)

                x_sil[jj_o] = x_sil[jj_o] + x[jj_i] * w
                y_sil[jj_o] = y_sil[jj_o] + y[jj_i] * w
                count = count+1

                last_index = jj_o[-1]

        x_sil = x_sil[0:last_index]
        y_sil = y_sil[0:last_index]

        return x_sil, y_sil

    def taa_corr(x, y):
        """ Returns correlation coefficient between column vectors x and y. """

        xn = x - np.mean(x, axis=0)
        xn = xn / np.sqrt(np.sum(np.power(xn, 2)))

        yn = y - np.mean(y, axis=0)
        yn = yn / np.sqrt(np.sum(np.power(yn, 2)))
        rho = np.sum(np.multiply(xn, yn))

        return rho
    print(len(x), len(y))
    assert len(x) == len(y), 'x and y should have the same length'

    fs = 10000
    N_frame = 256
    K = 512
    J = 15
    mn = 150
    H = third_oct(fs, K, J, mn)
    N = 30
    Beta = -15
    dyn_range = 40

    if fs_signal != fs:
        down_sampling = fs_signal / fs
        x = resample(x, int(len(x) / down_sampling))
        y = resample(y, int(len(y) / down_sampling))

    # --------------------------- Remove silent frames --------------------------- #

    x, y = remove_silent_frames(x, y, dyn_range, N_frame, N_frame / 2)

    # ------------------ Apply 1/3 octave band TF-decomposition ------------------ #

    x_hat = stdft(x, N_frame, N_frame / 2, K).T
    y_hat = stdft(y, N_frame, N_frame / 2, K).T

    X = np.zeros((J, x_hat.shape[1]))
    Y = np.zeros((J, y_hat.shape[1]))

    for i in range(X.shape[1]):
        X[:, i] = np.sqrt(np.dot(H,np.power(np.abs(x_hat[:, i]), 2)))
        Y[:, i] = np.sqrt(np.dot(H,np.power(np.abs(y_hat[:, i]), 2)))

    # ------------------- Obtain intermediate intelligibility ------------------- #

    d_interm = np.zeros((J, X.shape[1] - N))
    c = 10 ** (-Beta/20)

    for m in range(N, X.shape[1]):
        X_seg = X[:, (m-N):m]
        Y_seg = Y[:, (m-N):m]

        alpha = np.sqrt(np.divide(np.sum(np.power(X_seg, 2), axis=1), np.sum(np.power(Y_seg, 2), axis=1)))
        aY_seg = np.multiply(Y_seg, np.tile(alpha[:,None], (1, N)))

        for j in range(J):
            Y_prime = np.min([aY_seg[j, :], X_seg[j, :] + X_seg[j, :] * c],axis=0)
            d_interm[j, m-N] = taa_corr(X_seg[j, :].T, Y_prime)

    # ------------ Combine all intermediate intelligibility measures ------------ #

    return np.mean(d_interm)


if __name__ == '__main__':
    sr_o, o = wavread('stoi_test/kh4_1_orig_audio.wav')
    sr_r, r = wavread('stoi_test/testkh4_1.wav')
    print(stoi(o, r[0:len(o)], sr_o))
