# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 10:53:56 2023

@author: sb3682
"""
import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append("../")
import numpy.matlib
import scipy.io as sio
import torch
import util
import scipy.signal as signal
import h5py
import time

# mat = sio.loadmat("D:/110 ASL Data/Raw1D/class_15_Alperen_3.mat")
# RPExt_2 = mat["data1D_cut"]

data = h5py.File("C:/Users/sb3682.ECE-2V7QHQ3/My Stuff/end-to-end model for HAR/TFA-Net-main/TF_sabya/generate_dataset/train_data.h5", 'r')
x = np.asarray(data['signal_clean_train'])
xc = np.asarray(data['signal_noise_train'])
x = x[6]
RPExt_2 = x[0] + 1j*x[1]
RPExt_2 = RPExt_2[None]

# fpass = 2000

# # Design the highpass filter
# b, a = signal.butter(4, fpass, 'high', fs=7000000)

# for i in range(len(RPExt_2)):
#     RPExt_2[i, :] = signal.lfilter(b, a, RPExt_2[i, :])
    
rd_window_size = 256
hop_size = 10
low_lim = -100
num_frames = len(RPExt_2[0])
loop = 1 + int((num_frames - rd_window_size) / hop_size)
spec_n = np.zeros((loop, 256), dtype=np.complex64)
# spectrogram[n] = spec_n.T.astype(np.float32)
start_time = time.time()
for i in range(loop):
    start = i * hop_size
    end = start + rd_window_size
    frame = RPExt_2[0][start:end] * np.hanning(rd_window_size)
    spec_n[i] = np.fft.fftshift(np.fft.fft(frame,256))
end_time = time.time()
time_spect = end_time-start_time
plt.figure(1)
plt.subplot(2,1,1)
plt.imshow(20*np.log((abs(spec_n.T)/np.max(abs(spec_n)))), cmap='jet',aspect='auto')
# plt.ylim(40, 160)
plt.colorbar()
plt.clim(0,low_lim)
plt.axis('off')
plt.margins(0, 0)

fr_path = 'C:/Users/sb3682.ECE-2V7QHQ3/My Stuff/end-to-end model for HAR/TFA-Net-main/TF_sabya/checkpoint/experiment_name/fr/epoch_10.pth'
# fr_path = 'C:/Users/sb3682.ECE-2V7QHQ3/My Stuff/end-to-end model for HAR/DeepFreq-master/DeepFreq-main/DeepFreq-master/checkpoint/experiment_name/fr/epoch_10.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#load models
fr_module, _, _, _, _ = util.load(fr_path, 'fr', device)
fr_module.cpu()
fr_module.eval()

# x = (x/np.sqrt(np.mean(np.power(x, 2)))).astype(np.float32())
x = x.astype(np.float32())
start_time = time.time()

with torch.no_grad():
    RD_clean, RD_fr = fr_module(torch.tensor(x[None]))
   
end_time = time.time()
time_unet = end_time-start_time
RD_fr = np.squeeze(RD_fr, axis=0)
RD_fr = np.squeeze(RD_fr, axis=0)
RD_fr = RD_fr.numpy()
RD_fr[RD_fr < 0.01] = 0.0001
plt.subplot(2,1,2)
plt.imshow(20*np.log(RD_fr/np.max(RD_fr)), cmap='jet',aspect='auto')
# plt.ylim(200, 800)
#plt.ylim(-1500,2500)
plt.colorbar()
plt.clim(0,low_lim)
plt.axis('off')
plt.margins(0, 0)