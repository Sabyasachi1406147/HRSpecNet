# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 12:02:53 2023

@author: sb3682
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import cv2
from scipy.signal import savgol_filter
import random
import h5py

sampling_freq = 3200
num_samples = 1
num_freq = 1
nfreq = np.random.randint(1, num_freq + 1, num_samples)
norm_freq1 = (np.random.rand(num_samples, num_freq)-0.5)*20*(sampling_freq/8000) #12
norm_freq2 = (np.random.rand(num_samples, num_freq)-0.5)*12*(sampling_freq/8000) #8
norm_freq3 = (np.random.rand(num_samples, num_freq)-0.5)*15000*(sampling_freq/8000)
theta = np.random.rand(num_samples, num_freq) * 2 * np.pi

# training snr
snr = np.array([20, 5, 8, 10, 12, 15, 20])

# # testing snr
# snr = np.array([5, 5, 5, 5, 5, 5])
# snr = np.array([10, 10, 10, 10, 10, 10])
# snr = np.array([15, 15, 15, 15, 15, 15])
# snr = np.array([20, 20, 20, 20, 20, 20])
# snr = np.array([0, 0, 0, 0, 0, 0])

snr = 10**(snr/10)

duration = 1.5 # in sec
signal_dim = int(sampling_freq*duration) # signal_length

xgrid_t = np.linspace(0, duration, signal_dim)

# xgrid_t_down = np.linspace(0, duration, signal_dim_down)

amplitude_fr = (np.random.rand(num_samples, num_freq))*(40-0.5) + 0.5
amplitude_fr = amplitude_fr/norm_freq1
amplitude_td = 8*np.random.rand(num_samples, num_freq)+1 #8

signal_noise = np.zeros((num_samples, 2, signal_dim),dtype=np.float16)
signal_clean = np.zeros((num_samples, 2, signal_dim),dtype=np.float16)
actual_freq = np.zeros((num_samples, num_freq, signal_dim))

# Step 2: Define STFT parameters
window_size = 256
hop_size = 10 #int((window_size-((np.floor(window_size/100))*100))/2)
ygrid_f = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
# Step 3: Calculate the number of frames and create an array to store the spectrogram
num_frames = 1 + int((signal_dim - window_size) / hop_size)
spectrogram = np.zeros((num_samples, window_size, num_frames),dtype=np.float32)

window_size_down = window_size

# fr = np.zeros((num_samples, window_size, signal_dim),dtype=np.float16)

fr_down = np.zeros((num_samples, window_size, num_frames),dtype=np.float16)

down_f = int(window_size/window_size_down)

temp = np.linspace(-0.5,0.5,window_size) #array of 256 ranged from -0.5 to 0.5

temp_down = np.linspace(-0.5,0.5,window_size_down)

n_complex = np.random.randn(signal_dim, 2)
n = 0

for n in range(int(num_samples)):
    snr_random = random.choice(snr)
    if n % 100 == 0:
        print(f"sample: {n}")
    for i in range(nfreq[n]):
        freq_wave =  2 * np.pi * (norm_freq1[n,i] + (norm_freq2[n,i]*2*xgrid_t)) * amplitude_fr[n,i] * np.cos((2 * np.pi * (norm_freq1[n,i] * xgrid_t + norm_freq2[n,i] * (xgrid_t**2))) + theta[n,i])+ norm_freq3[n,i]
        actual_freq[n,i,:] = (2 * np.pi * (norm_freq1[n,i] + (norm_freq2[n,i]*2*xgrid_t)) * amplitude_fr[n,i] * np.cos((2 * np.pi * (norm_freq1[n,i] * xgrid_t + norm_freq2[n,i] * (xgrid_t**2))) + theta[n,i])+ norm_freq3[n,i])/sampling_freq
        freq = amplitude_fr[n,i] * np.sin(2 * np.pi * (norm_freq1[n,i]*xgrid_t + norm_freq2[n,i] * (xgrid_t**2)) + theta[n,i]) + (norm_freq3[n,i] * xgrid_t)
        if np.max(np.abs(actual_freq[n,i,:])) > 0.5:
            factor = 1.5*2*np.max(np.abs(actual_freq[n,i,:])) #1.5
            actual_freq[n,i,:] = actual_freq[n,i,:]/factor
            freq = freq/factor
        
        sin = amplitude_td[n,i]*np.exp(2j*np.pi*freq)
        signal_clean[n][0] = signal_clean[n][0] + sin.real[None]
        signal_clean[n][1] = signal_clean[n][1] + sin.imag[None]
    
    signal_clean[n] = signal_clean[n] / (np.sqrt(np.mean(np.power(signal_clean[n], 2)))+1e-10)
    n_complex = n_complex / np.sqrt((np.mean((np.abs(n_complex))**2))*(10**(snr_random/10)))
    signal_noise[n] = signal_clean[n] + n_complex.T
    
    samp_freq_down = np.zeros((nfreq[n], num_frames))
    samp_freq_main = actual_freq[n,:nfreq[n]]
    
    # samp_freq = samp_freq_main
    # # mask = samp_freq_main > 0.5 #if goes above 0.5 alias back. 0.6 will go back to (0.6-1)=-0.4
    # # samp_freq[mask] -= 1 
    # # mask = samp_freq < -0.5
    # # samp_freq[mask] += 1
    # # mask = samp_freq > 0.5 #2nd one if it goes to 1.9 first comes back to 0.9 then to -0.1
    # # samp_freq[mask] -= 1 
    # # mask = samp_freq < -0.5
    # # samp_freq[mask] += 1 
    # freq_amplitude = (np.ones((len(samp_freq),signal_dim)))*(np.tile(amplitude_td[n,:nfreq[n]],(signal_dim,1)).T)
    # # this is the amplitude of each frequency component reshaped to use in the fr
    # index_values = np.searchsorted(temp, samp_freq) # find out the index of each freq components
    # matrix_2d = np.zeros((window_size, signal_dim))
    # matrix_2d[index_values, np.arange(signal_dim)] = freq_amplitude # put the indexed values in the 2D matrix
    # kernel_size = (5,1)  # Choose the kernel size (odd numbers)
    # sigma_x = 0  # Standard deviation in X direction (0 means calculated from kernel size)
    # fr[n] = (cv2.GaussianBlur(matrix_2d, kernel_size, sigma_x)).astype(np.float16) # using a gausssian kernel for smoothing
    

    for i in range(num_frames):
        start = i * hop_size
        end = start + window_size
        down = np.mean(samp_freq_main[:,start:end],axis=1)
        samp_freq_down[:,i] = down
    # downsample = int(np.ceil(signal_dim/num_frames))
    # new_dim = int(num_frames*downsample)
    # diff = int((new_dim-signal_dim))
    
    # b = np.zeros((nfreq[n],new_dim))
    # b[:,:signal_dim] = samp_freq_main
    # samp_freq_down[:] = b[:,::downsample]
    
    freq_amplitude_down = (np.ones((len(samp_freq_main),len(samp_freq_down[0]))))*(np.tile(amplitude_td[n,:nfreq[n]],(len(samp_freq_down[0]),1)).T)
    index_values_down = np.searchsorted(temp_down, samp_freq_down) # find out the index of each freq components
    matrix_2d_down = np.zeros((window_size_down, len(samp_freq_down[0])))
    matrix_2d_down[index_values_down, np.arange(len(samp_freq_down[0]))] = freq_amplitude_down # put the indexed values in the 2D matrix
    
    kernel_size_down = (1,3)  # Choose the kernel size (odd numbers)
    sigma_x = 0  # Standard deviation in X direction (0 means calculated from kernel size)
    fr_down[n] = cv2.GaussianBlur(matrix_2d_down, kernel_size_down, sigma_x) # using a gausssian kernel for smoothing       
        
    # fr_down[n] = matrix_2d_down.astype(np.float16)
        
    spec_n = np.zeros((num_frames, window_size), dtype=np.complex64)
    
    # Step 4: Compute the STFT using np.fft.fft
    signal_com = signal_clean[n][0] + signal_clean[n][1]*1j
    for i in range(num_frames):
        start = i * hop_size
        end = start + window_size
        frame = signal_com[start:end] * np.hanning(window_size)
        spec_n[i] = np.fft.fftshift(np.fft.fft(frame))
    
    # Step 5: Calculate the magnitude of the STFT
    spec_n = np.abs(spec_n)
    spectrogram[n] = spec_n.T.astype(np.float32)

ex = 0

# Step 6: Display the spectrogram
plt.figure(figsize=(10, 6))
plt.imshow(20 * np.log10(spectrogram[ex]/np.max(spectrogram[ex])), aspect='auto', cmap='jet', origin='lower')
y_tick_positions = np.linspace(0, spectrogram[ex].shape[0]-1, 11).astype(int)
plt.yticks(y_tick_positions, ygrid_f)
plt.colorbar(format='%+2.0f dB')
plt.clim(-60,0)
# plt.xlabel('Time Frame')
plt.ylabel('Frequency Bin')
plt.title('Spectrogram')
plt.show()

fr_down[ex][fr_down[ex] < 0.001] = 0.0001
plt.figure(figsize=(10, 6))
plt.imshow(20*np.log(fr_down[ex]), aspect='auto', cmap='jet', origin='lower')
y_tick_positions = np.linspace(0, fr_down[ex].shape[0]-1, 11).astype(int)
plt.yticks(y_tick_positions, ygrid_f)
plt.colorbar(format='%+2.0f dB')
plt.title('Ground Truth')
plt.clim(-20,0)


import matplotlib.pyplot as plt
import cv2
import random
import util_v2
import torch
import time

fr_path = 'C:/Users/sb3682.ECE-2V7QHQ3/My Stuff/end-to-end model for HAR/TFA-Net-main/TF_sabya/checkpoint/experiment_name/fr/epoch_10.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#load models
fr_module, _, _, _, _ = util_v2.load(fr_path, 'fr', device)
fr_module.cpu()
fr_module.eval()

start_time = time.time()
with torch.no_grad():
    cae, c_re, RD_fr = fr_module(torch.tensor(signal_noise.astype(np.float32())))
end_time = time.time()

cae = cae.numpy()

time = (end_time-start_time)/num_samples

RD_fr = np.squeeze(RD_fr, axis=0)
RD_fr = np.squeeze(RD_fr, axis=0)
RD_fr = RD_fr.numpy()
RD_fr[RD_fr < 0.0001] = 0.00001
plt.figure(figsize=(10, 6))
plt.imshow((20*np.log(RD_fr/np.max(RD_fr))), cmap='jet',aspect='auto')
y_tick_positions = np.linspace(0, RD_fr.shape[0]-1, len(ygrid_f)).astype(int)
plt.yticks(y_tick_positions, ygrid_f)
plt.ylim(0, 256)
plt.colorbar(format='%+2.0f dB')
plt.clim(-60,0)
plt.title('Auto-UNet', fontname='Times New Roman', fontsize=24)
plt.axis('off')
plt.margins(0, 0)
# hf = h5py.File('C:/Users/sb3682.ECE-2V7QHQ3/My Stuff/end-to-end model for HAR/TFA-Net-main/TF_sabya/generate_dataset/train_data.h5', 'w')
# hf.create_dataset('signal_clean_train', data=signal_clean)
# hf.create_dataset('signal_noise_train', data=signal_noise)
# hf.create_dataset('freq_train', data=fr_down)
# # hf.create_dataset('spect', data=spectrogram)
# hf.close()

# hf = h5py.File('C:/Users/sb3682.ECE-2V7QHQ3/My Stuff/end-to-end model for HAR/TFA-Net-main/TF_sabya/generate_dataset/test_10db.h5', 'w')
# hf.create_dataset('signal_clean_test', data=signal_clean)
# hf.create_dataset('signal_noise_test', data=signal_noise)
# hf.create_dataset('freq_test', data=fr_down)
# # hf.create_dataset('spect', data=spectrogram)
# hf.close()