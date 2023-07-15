import pickle
import mne
import matplotlib.pyplot as plt
from preprocess import *
import numpy as np
from convert_data import *

path = 'BETA Dataset/pickle/S1.pkl'
with open(path, 'rb') as f:
    data = pickle.load(f)

EEG_data = data['EEG']

info = mne.create_info(
    ch_names=['PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8', 'O1', 'Oz', 'O2'],
    ch_types=['eeg'] * len(['POz', 'PO3', 'PO4', 'PO5', 'PO6', 'PO7', 'PO8', 'Oz', 'O1', 'O2']),
    sfreq=250
)  # 设置取自“世界机器人大赛”

ch_number = [53 - 1, 54 - 1, 55 - 1, 56 - 1, 57 - 1, 58 - 1, 59 - 1, 61 - 1, 62 - 1, 63 - 1]  # 设置取自”loc“,用MATLAB查看

Op_EEG = EEG_data[:, :, 0, 0][ch_number]

Op_raw = mne.io.RawArray(Op_EEG / 50, info)

scalings = {'eeg': 1, 'grad': 1, 'eog': 2}
Op_raw.plot(n_channels=len(ch_number), scalings=scalings, title='my', show=True,
            block=True)
plt.show()

path = 'course data/S1/block1.mat'
trial = get_data_from_mat(path)[0]
trial_raw = mne.io.RawArray(trial / 500, info)
trial_raw.plot_psd(average=True)
trial = band_pass(trial, 1, 120, 250, 6)
trial_raw = mne.io.RawArray(trial / 500, info)
trial_raw.plot_psd(average=True)

filter_data = filt_data(Op_EEG, 8, 88, 250, 6)
Op_raw = mne.io.RawArray(filter_data / 50, info)
Op_raw.plot(n_channels=len(ch_number), scalings=scalings, title='my', show=True,
            block=True)
plt.show()

i = 0
for eeg in filter_data:
    f = np.fft.fft(eeg)
    f = np.abs(f)
    if i == 0:
        f0 = f
    else:
        f0 += f
    i += 1
f0 /= i
plt.plot(f0)
plt.show()

locs_info_path = 'Benchmark Dataset/64-channels.loc'
montage = mne.channels.read_custom_montage(locs_info_path)
chan_names = np.loadtxt(locs_info_path, dtype=str, usecols=3)
info = mne.create_info(
    ch_names=chan_names.tolist(),
    ch_types=['eeg'] * 64,
    sfreq=250
)  # 设置取自“世界机器人大赛”
Op_EEG = EEG_data[:, :, 0, 0]
Op_EEG[ch_number] = 0.00001
Op_raw = mne.io.RawArray(Op_EEG, info)
Op_raw.set_montage(montage)
Op_raw.plot_sensors(ch_type='eeg', show_names=True)
Op_raw.plot_psd_topo()
