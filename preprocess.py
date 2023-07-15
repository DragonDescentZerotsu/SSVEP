import scipy.signal as signal


def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], 'bandpass')
    return b, a


def filt_data(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    filter_data = signal.filtfilt(b, a, data, axis=-1, padtype='odd', padlen=None, method='pad', irlen=None)
    return filter_data


def preprocess(data, lowcut, highcut, fs, order):
    """截掉了开头cue的0.5s和视觉反应的0.14s,休息时间里保留了0.14s"""
    ch_number = [53 - 1, 54 - 1, 55 - 1, 56 - 1, 57 - 1, 58 - 1, 59 - 1, 61 - 1, 62 - 1, 63 - 1]  # 设置取自”loc“,用MATLAB查看
    Op_EEG = data[ch_number]
    filter_data = filt_data(Op_EEG, lowcut, highcut, fs, order)
    Op_EEG_flicker_filter = filter_data[:, int((0.5 + 0.14) * 250):]  # int((0.5 + 2 + 0.14) * 250)]
    return Op_EEG_flicker_filter


def band_pass(data, lowcut, highcut, fs, order):
    """截掉了trial开始时的0.14s反应时间"""
    filter_data = filt_data(data, lowcut, highcut, fs, order)
    f0 = 50
    Q = 30
    b, a = signal.iirnotch(f0, Q, fs)
    filter_data = signal.filtfilt(b, a, filter_data)
    # filter_data = filter_data[:, int(0.14 * 250):]
    # filter_data = signal.resample(filter_data, int(filter_data.shape[1] * 250 / 1000), axis=-1)
    filter_data = filter_data[:, int(0.14 * fs):]
    return filter_data
