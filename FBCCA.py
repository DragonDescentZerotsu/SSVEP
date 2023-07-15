from CCA import *
from preprocess import *
import numpy as np


def get_sub_bands(data, N, lowcut, highcut, fs, order, course=False):
    """N为sub-band数目,bandpass和preprocess中都去掉了开头的0.14s数据"""
    sub_bands_data = []
    if course:
        for i in range(N):
            sub_bands_data.append(band_pass(data, lowcut + i * 8 - 2, highcut + 2, fs, order))  # 每个sub-band留2Hz裕量
    else:
        for i in range(N):
            sub_bands_data.append(preprocess(data, lowcut + i * 8 - 2, highcut + 2, fs, order))  # 每个sub-band留2Hz裕量
    return np.array(sub_bands_data)


def get_w(a, b, N):
    return np.array([(n + 1) ** (-a) + b for n in range(N)])


def FBCCA_analyse(data, N, lowcut, highcut, f_interval, fs, order, Yf_all, w, k_range, course=False):
    """返回最有可能的f"""
    sub_bands_data = get_sub_bands(data, N=N, lowcut=lowcut, highcut=highcut, fs=fs, order=order, course=course)
    p_corr_bar = 0
    f_corr = 0
    for k in range(k_range):  # k=0~39
        f = 8 + f_interval * k
        Yf = Yf_all[k]
        n = 0  # 用于索引权重w
        p_k_bar = 0
        for sub_band_n in sub_bands_data:
            p_k_n = cca_analyse(sub_band_n, Yf)
            p_k_bar += w[n] * (p_k_n ** 2)
            n += 1
        if p_k_bar > p_corr_bar:
            p_corr_bar = p_k_bar
            f_corr = f
    return f_corr


if __name__ == '__main__':
    accuracy = []
    a = 1.25
    b = 0.25
    N = 3  # sub-band数量
    w = get_w(a, b, N)
    print('w got, start FBCCA')
    for subject in range(10):
        path = 'BETA Dataset/pickle/S{}.pkl'.format(subject + 1)
        with open(path, 'rb') as fb:
            data = pickle.load(fb)

        EEG = data['EEG']
        freqs = data['freqs']

        S = EEG.shape[1]  # 采样点数
        trail_total_number = EEG.shape[2] * EEG.shape[3]  # blocks * conditions = 160
        fs = 250  # 采样率250
        Nh = 6
        S = S - (0.5 + 0.14) * 250  # - (0.5 - 0.14) * 250  # 提示时间+视觉反应时间，来源“BETA”+“世界机器人大赛”

        Yf_all = get_Yf_all(int(S), 8, 15.8, 0.2, fs, Nh)  # 得到所有的参考矩阵后面直接查表加快运算速度

        subject_correct = 0
        for block in range(EEG.shape[2]):
            block_correct = 0
            for trial in range(EEG.shape[3]):
                EEG_data = EEG[:, :, block, trial]
                f_corr = FBCCA_analyse(EEG_data, N=N, lowcut=8, highcut=41, f_interval=0.2, fs=fs, order=6,
                                       Yf_all=Yf_all,
                                       w=w, k_range=40)
                if f_corr == freqs[trial]:
                    block_correct += 1
            subject_correct += block_correct
            print('subject {}: block {} accuracy:{}'.format(subject + 1, block + 1, block_correct / 40))
        accuracy.append(subject_correct / trail_total_number)

    print(accuracy)
    with open('BETA Dataset/FBCCA_accuracy_10_subjects.csv', 'w', newline='') as fb:
        writer = csv.writer(fb)
        for item in accuracy:
            writer.writerow([item])
