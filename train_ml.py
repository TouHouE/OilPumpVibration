from sklearn.ensemble import GradientBoostingClassifier
import joblib
from scipy.fft import fft, fftfreq
import os
import pandas as pd
import numpy as np
SAMPLE_RATE = 3e3

def get_freq_domain(time_domain):
    N = len(time_domain)
    intensity = fft(time_domain.values)  # intensity
    freq = fftfreq(N, 1 / SAMPLE_RATE)
    return np.abs(intensity)[:N // 2], freq[:N // 2]


def get_data(phase):
    root = './static/data'
    normal = pd.read_csv(f'{root}/Normal_{phase}.csv')
    error = pd.read_csv(f'{root}/Error_{phase}.csv')
    X_train, Y_train = [], []

    for index in range(len(normal)):
        intensity, _ = get_freq_domain(normal.iloc[index])
        X_train.append(intensity)
        Y_train.append(1)

    for index in range(len(error)):
        intensity, _ = get_freq_domain(error.iloc[index])
        X_train.append(intensity)
        Y_train.append(0)

    return np.array(X_train), np.array(Y_train)


def main():
    X_train, Y_train = get_data('train')
    model = GradientBoostingClassifier(n_estimators=1000, max_features='sqrt')
    model.fit(X_train, Y_train)
    X_test, Y_test = get_data('test')
    matrix = np.array([[0, 0], [0, 0]])
    pred = model.predict(X_test)

    for p, l in zip(pred, Y_test):
        matrix[l, p] += 1

    print(matrix)

main()

