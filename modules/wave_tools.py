import numpy as np
from glob import glob
from func_PyCC import *
from glob import glob
import h5py
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import copy
from scipy import signal
import math
import dascore as dc
from scipy.signal import find_peaks
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import savgol_filter
from scipy.signal import spectrogram

def fk(data, dx, dt):
    # From Ariel's repo
    (nch, nt) = np.shape(data)
    nf = 2 ** (1 + math.ceil(math.log(nt, 2)))
    nk = 2 ** (1 + math.ceil(math.log(nch, 2)))

    fft_f = np.arange(-nf / 2, nf / 2) / nf / dt
    fft_k = np.arange(-nk / 2, nk / 2) / nk / dx

    fk_res = np.fft.fftshift(np.fft.fft2(data, s=[nk, nf]))
    fk_res = np.absolute(fk_res)

    return fk_res, fft_f, fft_k
    
def map_fv(data, dx, dt, freqs, vels, norm=False):
    nscanv = len(vels)
    nscanf = len(freqs)

    if norm:
        data = data / np.linalg.norm(data, axis=-1, keepdims=True, ord=1)

    fk_res, fft_f, fft_k = fk(data, dx, dt)

    interp_fun = RegularGridInterpolator(
        (fft_k, fft_f),
        fk_res,
        bounds_error=False,
        fill_value=0.0
    )

    fv_map = np.zeros((nscanf, nscanv), dtype=np.float32)
    for i, fr in enumerate(freqs):
        ks = fr / np.array(vels)  # compute wavenumbers
        points = np.stack([ks, np.full_like(ks, fr)], axis=-1)
        fv_map[i, :] = interp_fun(points)

    # Smooth over frequency axis
    fv_map = savgol_filter(fv_map, 25, 4, axis=0)
    return fv_map.T  # shape: [vels x freqs]
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs                    # Nyquist Frequency
    normal_cutoff = cutoff / nyq      # Normalize cutoff
    sos = signal.butter(order, normal_cutoff, btype='low', analog=False,output='sos')
    return sos

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = signal.butter(order, normal_cutoff, btype='high', analog=False,output='sos')
    return sos

def butter_lowpass_filter(data, cutoff, fs, order=5):
    sos = butter_lowpass(cutoff, fs, order)
    y = signal.sosfiltfilt(sos, data)   # Zero-phase filtering
    return y

def butter_highpass_filter(data, cutoff, fs, order=5):
    sos = butter_highpass(cutoff, fs, order)
    y = signal.sosfiltfilt(sos, data)
    return y
def butter_bandpass_filter(data,f1,f2,fs,order=5):
    nyq = 0.5 * fs                   # Nyquist frequency
    low = f1 / nyq                  # Normalized low cutoff
    high = f2 / nyq                 # Normalized high cutoff
    sos = signal.butter(order, [low, high], btype='band', output='sos')
    y = signal.sosfiltfilt(sos,data)
    return y
def preprocess(x, fs, f1, f2, window_size):
    b = np.convolve(x, np.ones(window_size)/window_size, mode='same')
    c = b[window_size:-window_size]-x[window_size:-window_size]
    d = butter_bandpass_filter(c,f1,f2,fs)
    e = (d-np.mean(d))/np.std(d)
    e = e.astype('float32')
    return e


def interferometry(phase,fs,f1,f2,window_size,lag):
    phase2 = copy.copy(phase[:,window_size:-window_size])
    for i in range(phase2.shape[0]):
        phase2[i,:] = preprocess(phase[i,:], fs, f1, f2, window_size)
    a = phase2[100,:]
    peaks,_=find_peaks(a,prominence=10,distance=250)
    print(peaks)
    cc_final=np.empty((phase2.shape[0],2*lag*fs-1))
    for num,st in enumerate(peaks):
        if st+lag*fs<phase2.shape[1]:
            wn = int(lag*fs)
            data1 = phase2
            data = data1[:,st-wn:st+wn]
            nch = data.shape[0]
            npts = data.shape[1]
            pair_channel2 = list(range(0, nch, 1))      # Channels 1, 2, 3, ..., nch-1
            pair_channel1 = [100] * (len(pair_channel2))         # Always channel 0
            npair = len(pair_channel1)
            # print(f"Total pairs: {npair}")
              # Adjust based on your GPU memory
            n_total = len(pair_channel1)
            data = torch.from_numpy(data).to(torch.float32)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_conf = {
                "is_spectral_whitening": True,
                "whitening_params": [fs, 0, f1, f2],
            }
            model = Torch_cross_correlation(**model_conf)
            model = nn.DataParallel(model)
            model.to(device)
            model.eval()
            batch_size=20
            with torch.no_grad():
                data_ch = data
                # print(data_ch.shape)
                for b in range(0, n_total, batch_size):
                    b1 = pair_channel1[b:b+batch_size]
                    b2 = pair_channel2[b:b+batch_size]
                    data1 = data_ch[b1, :].reshape(-1, data.shape[1])
                    data2 = data_ch[b2, :].reshape(-1, data.shape[1])
                    cc = model(data2, data1)
                    cc = cc.cpu().numpy()
                    if b==0:
                        cc_batches=cc[:,100:-100]
                    else:
                        cc_batches = np.concatenate((cc_batches, cc[:,100:-100]), axis=0)
            if num==0:
                cc_final = cc_batches/np.max(np.abs(cc_batches),axis=1,keepdims=True)
            else:
                cc_final = cc_final+cc_batches/np.max(np.abs(cc_batches),axis=1,keepdims=True)
    return cc_final