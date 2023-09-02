import numpy as np
from scipy.signal import stft, lfilter
from scipy.fft import dct

class MFCC:
    def __init__(self, window_length, window_overlap, melbands, maxmel):
        self.win_len = window_length
        self.win_overlap = window_overlap
        self.melbands = melbands
        self.maxmel = maxmel

    def spectrum(self, x, fs, N, f_scale='Hz'):
        X = abs(np.fft.fft(x, N))
        X = X[0:N//2]
        f = np.fft.fftfreq(n = N, d = 1/fs)
        f = f[0:N//2]
        if f_scale == 'kHz':
            f = f/1000
        return X, f
    
    def pre_emphasis(self, y, fs):
        N = len(y)
        yf = lfilter(b=[1, -0.68], a=1, x=y)
        Yf, f = self.spectrum(yf, fs, N, f_scale='Hz')
        return yf, Yf, f

    def dft(self, y, fs, window_length, window_overlap):
        f, _, Sxx = stft(x=y,fs=fs,window='hann',nperseg=window_length,noverlap=window_overlap)
        return Sxx, f

    def freq2mel(self,f): 
        return 2595*np.log10(1 + (f/700))
    
    def mel2freq(self,m): 
        return 700*(10**(m/2595) - 1)

    def melbank(self, melbands, maxmel, f, spectrum_len):
        mel_idx = np.linspace(0,maxmel,melbands+2)
        freq_idx = self.mel2freq(mel_idx)
        k = f*1000
        melfilterbank = np.zeros((spectrum_len,melbands))

        for j in range(1,melbands):
            l_j = freq_idx[j-1]
            c_j = freq_idx[j]
            u_j = freq_idx[j+1]
            upslope = (k-l_j)/(c_j-l_j)
            downslope = (u_j-k)/(u_j-c_j)

            if j==1:
                upslope = 1.0 + 0*k
            if j==melbands-1:
                downslope = 1.0 + 0*k

            melfilterbank[:,j-1] = np.max([0*k,np.min([upslope,downslope],axis=0)],axis=0)

        melreconstruct = np.matmul(np.diag(np.sum(melfilterbank**2+1e-12,axis=0)**-1),np.transpose(melfilterbank))
        return melfilterbank, melreconstruct
    
    def logmelspectrogram(self, filter_bank, STFT):
        return np.log(np.matmul((np.abs(STFT)**2).T,filter_bank)+1e-12)

    
    def mfcc(self, y, fs):
        yf, _, _ = self.pre_emphasis(y,fs)
        Sx, _ = self.dft(yf, fs, self.win_len, self.win_overlap)
        melfilterbank, _ = self.melbank(self.melbands, self.maxmel)
        logmel = self.logmelspectrogram(melfilterbank, Sx)
        mfcc = dct(logmel)
        return mfcc