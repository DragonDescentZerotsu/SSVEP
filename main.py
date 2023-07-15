from convert_data import get_data_from_mat
from FBCCA import *
import numpy as np
import pandas as pd
from CCA import *

a = 1.25
b = 0.25
N = 7
fs = 250
Nh = 6  # harmonic谐波数量

w = get_w(a, b, N)

count_for_12_5_right = 0
count_for_12_5 = 0

csv_label_result = []
csv_f_result = []
columns = []
f_reference = [8 + 0.3 * i for i in range(20)]
f_reference[np.where(abs((np.array(f_reference) - 13.4)) < 0.1)[0][0]] = 13.4
print('\n-----  start  -----')
for subject in range(5):
    for block in range(2):
        path = 'course data/S{}/block{}.mat'.format(subject + 1, block + 1)
        trials = get_data_from_mat(path)
        block_f_result = []
        block_label_result = []
        for trial in trials:
            S = trial.shape[1]
            S = S - 0.14 * fs
            Yf_all = get_Yf_all(int(S), 8, 13.7, 0.3, fs, Nh)
            f_corr = FBCCA_analyse(trial, N=N, lowcut=8, highcut=88, f_interval=0.3, fs=fs, order=6, Yf_all=Yf_all,
                                   w=w, k_range=20, course=True)
            if f_corr == 12.5:
                f_corr = FBCCA_analyse(trial, N=3, lowcut=8, highcut=41, f_interval=0.3, fs=fs, order=6, Yf_all=Yf_all,
                                       w=w, k_range=20, course=True)
                if f_corr == CCA_course(trial, 6, 16, fs, 0.3, 20, Yf_all):
                    count_for_12_5_right += 1
                count_for_12_5 += 1
            if abs(f_corr - 13.4) < 0.1:  # 否则会变成13.3999999
                f_corr = 13.4
            block_f_result.append(f_corr)
            block_label_result.append(f_reference.index(f_corr) + 1)
        print('subject {}, block {}: '.format(subject + 1, block + 1) + str(block_f_result))
        csv_f_result.append(block_f_result)
        csv_label_result.append(block_label_result)
        columns.append('S{}block{}'.format(subject + 1, block + 1))

csv_f_result, csv_label_result = np.array(csv_f_result).T, np.array(csv_label_result).T

print('12.5Hz same rate: {}'.format(count_for_12_5_right / count_for_12_5))

f_count = []
for f in f_reference:
    f_count.append(len(np.where(csv_f_result == f)[0]))

print('\n-----  frequency count  -----')
print(f_reference)
print(f_count)
print('average: {}'.format(np.sum(np.array(f_count)) / len(f_count)))

f_count = np.array(f_count).reshape(1, -1)
csv_count = pd.DataFrame(f_count, columns=f_reference)
csv_count.to_csv('f_count_low8_high88_12.5.csv')  # 12.5Hz明显是异常值，有16个，远大于其他，应该有3~6个分错了

csv_f_result = pd.DataFrame(csv_f_result, columns=columns)
csv_label_result = pd.DataFrame(csv_label_result, columns=columns)
csv_f_result.to_csv('course_f_result_12.5.csv')
csv_label_result.to_csv('course_label_result_12.5.csv')
