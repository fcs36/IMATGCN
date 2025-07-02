import pandas as pd
import numpy as np
import os
import random



# plant anomaly
data2 = np.load('data/data1.npy')
def point_anomaly(data, k, star, end):   # spike
    data1 = data
    list = [-1, 1]
    lenth = data1.shape[0]
    ay = data1
    index = random.sample(range(star, end), 30)
    if end <= lenth and star >= 0:
        for i in range(len(index)):
            ay[index[i]] = ay[index[i]] + k*random.choice(list)
    data = np.vstack([ay,data1])
    return ay,index

def drift_anomaly(data, k, star, end):  # drift
    lenth = data.shape[0]
    ay = data
    index = np.arange(star, end)
    if end < lenth and star > 0:
        t = np.arange(end-star)
        c = k/(end-star)
        ay[star:end] = data[star:end] + c*t
    return ay,index

def bias_anomaly(data, k, star, end):   # bias
    lenth = data.shape[0]
    ay = data
    index = np.arange(star, end)
    if end < lenth and star > 0:
        t = np.arange(end - star)
        c = k / (end - star)
        ay[star:end] = data[star:end] + 0.3
    return ay, index

def normal_anomaly(data, k, star, end):     # static
    lenth = data.shape[0]
    ay = data
    index = np.arange(star, end)
    if end <= lenth and star > 0:
        ay[star:end] = k
    return ay, index



# Anomaly
# data2[0],index = point_anomaly(data2[0], 0.5, 18020, 27020)

# data2[0],index = drift_anomaly(data2[0], 0.5, 19500, 21500)

# data2[0],index = bias_anomaly(data2[0], 0.5, 19500, 21500)

# data2[0],index = normal_anomaly(data2[0], 2, 25500, 26020)

np.save('data/data2.npy', data2)


index = np.array(index)
ano_index = index - data2.shape[1]
test_size = 9000
ano_label = np.zeros(test_size)
ano_label[ano_index] = 1
np.save('data/ano_label.npy', ano_label)


data2 = np.load('data/data2.npy')
def add_gaussian_noise(signal, snr):
    signal_power = np.sum(signal ** 2) / len(signal)
    noise_power = signal_power / (10 ** (snr / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    noisy_signal = signal + noise
    return noisy_signal


snr = 40
noisy_data = np.zeros_like(data2)

for i in range(data2.shape[0]):
    noisy_data[i, :] = add_gaussian_noise(data2[i, :], snr)
data3 = noisy_data
np.save('data/data3.npy', data3)


data = np.load('data/data3.npy')
num_samples = data3.shape[0]
num_features = 20
window_size = 20

num_windows = 27000


data_win_x = np.zeros((num_windows, num_samples, num_features))
data_win_y = np.zeros((num_windows, num_samples))


for i in range(num_windows):
    start = i
    end = start + window_size
    window_data = data[:, start:end]
    data_win_x[i] = window_data[:, :]
    data_win_y[i] = data[:,end]

np.save('data/data_win_x.npy', data_win_x)
np.save('data/data_win_y.npy', data_win_y)



