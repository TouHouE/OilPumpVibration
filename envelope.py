from scipy.signal import hilbert, chirp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
SAMPLE_RATE = 3e3  # 3k Hz

def main():
    data = pd.read_csv('./static/data/Error_train.csv').iloc[0]
    time = np.arange(len(data)) / SAMPLE_RATE
    signal = chirp(time, 1.0, time[-1], 100.0)
    signal *= data.values

    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = (np.diff(instantaneous_phase) /
                               (2.0 * np.pi) * SAMPLE_RATE)

    fig, (ax0, ax1) = plt.subplots(nrows=2)
    ax0.plot(time, signal, label='signal')
    ax0.plot(time, amplitude_envelope, label='envelope')
    ax0.set_xlabel("time in seconds")
    ax0.legend()
    ax1.plot(time[1:], instantaneous_frequency)
    ax1.set_xlabel("time in seconds")
    # ax1.set_ylim(0.0, 120.0)
    fig.tight_layout()
    plt.show()
    pass




if __name__ == '__main__':
    main()