import pickle
import math
import numpy as np
import csv
from sklearn.cross_decomposition import CCA
from preprocess import *


def get_Yf(S, f, fs, Nh):  # 变量命名规则与“脑机接口中的机器学习”（Notability）相同，计算规则与“CCA_better”（PDF）也一致
    Yf = []
    for i in range(Nh):
        y_harmonic = [math.sin(2 * (j + 1) * math.pi * (i + 1) * f / fs) for j in range(S)]
        Yf.append(y_harmonic)
        y_harmonic = [math.cos(2 * (j + 1) * math.pi * (i + 1) * f / fs) for j in range(S)]
        Yf.append(y_harmonic)
    return np.array(Yf)


def get_Yf_all(S, f_start, f_end, f_interval, fs, Nh):
    """做好所有的参考矩阵，后面要用时可以直接查表减少运算时间"""
    Yf_all = []
    for i in range(int((f_end - f_start) / f_interval) + 1):
        Yf_all.append(get_Yf(S, f_start + f_interval * i, fs, Nh))
    return np.array(Yf_all)


def cca_analyse(data, Yf):
    """CCA求相关系数算法"""
    data, Yf = data.T, Yf.T  # 从'https://scikit-learn.org.cn/view/460.html'可见fit()输入形状为[sample, feature]
    cca = CCA(n_components=1)
    X_train_r, Y_train_r = cca.fit_transform(data, Yf)
    p = np.corrcoef(X_train_r[:, 0], Y_train_r[:, 0])[0, 1]
    return p


def CCA_course(data, lowcut, highcut, fs, f_interval, k_range, Yf_all):
    preprocessed_data = band_pass(data, lowcut, highcut, fs, 6)
    p_corr = 0
    f_corr = 0
    for k in range(k_range):  # k=0~39
        f = 8 + f_interval * k
        Yf = Yf_all[k]
        p = cca_analyse(preprocessed_data, Yf)
        if p > p_corr:
            p_corr = p
            f_corr = f
    return f_corr


if __name__ == '__main__':
    accuracy = []
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
        S = S - (0.5 + 0.14) * 250 #- (0.5 - 0.14) * 250  # 提示时间+视觉反应时间，来源“BETA”+“世界机器人大赛”

        Yf_all = get_Yf_all(int(S), 8, 15.8, 0.2, fs, Nh)  # 得到所有的参考矩阵后面直接查表加快运算速度

        subject_correct = 0
        for block in range(EEG.shape[2]):
            block_correct = 0
            for trial in range(EEG.shape[3]):
                EEG_data = EEG[:, :, block, trial]
                preprocessed_data = preprocess(EEG_data, 6, 18, fs, 6)
                p_corr = 0
                f_corr = 0
                for k in range(40):  # k=0~39
                    f = 8 + 0.2 * k
                    Yf = Yf_all[k]
                    p = cca_analyse(preprocessed_data, Yf)
                    if p > p_corr:
                        p_corr = p
                        f_corr = f
                if f_corr == freqs[trial]:
                    block_correct += 1
            subject_correct += block_correct
            print('subject {}: block {} accuracy:{}'.format(subject + 1, block + 1, block_correct / 40))
        accuracy.append(subject_correct / trail_total_number)

    print(accuracy)
    with open('BETA Dataset/CCA_accuracy_10_subjects_BETA_band_extended.csv', 'w', newline='') as fb:
        writer = csv.writer(fb)
        for item in accuracy:
            writer.writerow([item])
