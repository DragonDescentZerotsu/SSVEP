import scipy.io as io
from scipy import signal
import pickle
import numpy as np


def get_data_from_mat(path):
    """取得trials并下采样"""
    trials = []
    raw_data = io.loadmat(path)['data']
    trial_starts = np.argwhere(raw_data[-1] == 1)
    trial_ends = np.argwhere(raw_data[-1] == 241)
    for i in range(trial_starts.shape[0]):
        trial = raw_data[:10, trial_starts[i][0]:trial_ends[i][0]]
        # trials.append(trial)
        down_sampled = signal.resample(trial, int(trial.shape[1] * 250 / 1000), axis=-1)
        trials.append(down_sampled)
    return trials


if __name__ == '__main__':
    subject = 1
    path = 'course data/S1/block1.mat'  # .format(subject)

    # for test:
    trials = get_data_from_mat(path)

    data = io.loadmat(path)
    EEG_data = data['data'][0, 0]['EEG']
    freqs = data['data'][0, 0]['suppl_info'][0, 0]['freqs'][0]
    save_data = {'EEG': EEG_data, 'freqs': freqs}

    with open('BETA Dataset/S{}.pkl'.format(subject), 'wb') as fb:
        pickle.dump(save_data, fb)

    # with open('BETA Dataset/S{}.pkl'.format(subject), 'rb') as f:
    #     data_show = pickle.load(f)
    # print(data_show)
