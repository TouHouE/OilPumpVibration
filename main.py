from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq
SAMPLE_RATE = 3e3


def make_as_image():
    all_normal = pd.read_csv('./static/data/Normal_test.csv')
    ax = plt.figure()
    ax.gca().set_axis_off()
    for index in range(len(all_normal)):
        data = all_normal.iloc[index]
        f, t, Sxx = signal.spectrogram(data.values, SAMPLE_RATE)
        plt.pcolormesh(t, f, Sxx, shading='gouraud')
        ax.savefig(f'./static/test/normal/{index + 1}.png', bbox_inches='tight', pad_inches=0)


def spectrum_main():
    data = pd.read_csv('./static/data/Normal_train.csv')
    plt.figure(figsize=(20, 15))
    for index in range(len(data)):
        if index == 5:
            break
        f, t, Sxx = signal.spectrogram(data.iloc[index].values, SAMPLE_RATE)
        plt.subplot(2, 5, index + 1)
        plt.pcolormesh(t, f, Sxx, shading='gouraud')
        plt.title('Normal')
    error_data = pd.read_csv('./static/data/Error_train.csv')

    for index_error in range(len(error_data)):
        if index_error == 5:
            break
        f, t, Sxx = signal.spectrogram(error_data.iloc[index_error].values, SAMPLE_RATE)
        plt.subplot(2, 5, index + index_error + 1)
        plt.pcolormesh(t, f, Sxx, shading='gouraud')
        plt.title(f'Error line : {index_error}')
    plt.show()

def main():
    def get_freq_domain(time_domain):
        N = len(time_domain)
        intensity = fft(time_domain.values)  # intensity
        freq = fftfreq(N, 1 / SAMPLE_RATE)
        return np.abs(intensity)[:N // 2], freq[:N // 2]
    plt.figure(figsize=(20, 15))

    normal = pd.read_csv('./static/data/Normal_train.csv')
    error = pd.read_csv('./static/data/Error_train.csv')

    for index in range(len(normal)):
        if index == 5:
            break
        intensity, freq = get_freq_domain(normal.iloc[index])
        plt.subplot(2, 5, index + 1)
        plt.plot(freq, intensity)
        plt.title(f'Normal Line: {index + 1}')

    for index_ in range(len(error)):
        if index_ == 5:
            break
        intensity, freq = get_freq_domain(error.iloc[index_])
        plt.subplot(2, 5, index + index_ + 1)
        plt.plot(freq, intensity)
        plt.title(f'Error Line: {index_ + 1}')
    plt.show()



# main()
# spectrum_main()
# make_as_image()
