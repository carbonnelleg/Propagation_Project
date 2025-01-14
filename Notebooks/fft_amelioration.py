import numpy as np
from scipy import fft
from matplotlib import pyplot as plt

t = np.arange(1000)
a = .2*np.sin(t/50)\
    + .08*np.sin(t/8+3) + .06*np.sin(t/7-4) + .033*np.sin(t/6+6) + .02*np.sin(t/5+8) \
    + .08*np.sin(t/4+3) + .06*np.sin(t/3.3-4) + .033 * \
    np.sin(t/3+6) + .02*np.sin(t/2.6+8)

extend_lens = [1, 100, 200, 300]
for extend_len in extend_lens:
    longer_a = np.concatenate((a[::-1][-extend_len:], a, a[::-1][:extend_len]))
    longer_t = np.arange(len(longer_a))-extend_len
    l = len(longer_t)

    A = fft.fft(longer_a, norm='forward')
    w = fft.fftfreq(len(longer_t))

    freq_filt_param = 200

    half_win_width = l//freq_filt_param+1
    window = np.zeros(l, dtype=float)
    window[:half_win_width] = 1.0
    window[len(window)-(half_win_width-1):] = 1.0
    A_filt = A*window

    a_filt = np.real(fft.ifft(A_filt, norm='forward'))[extend_len:-extend_len]

    fig, (ax1, ax2) = plt.subplots(2, 1, height_ratios=[2, 1])
    ax1.plot(longer_t, longer_a, color='lightblue')
    ax1.plot(t, a, color='blue')
    ax1.plot(t, a_filt, color='black')
    ax2.plot(np.concatenate([w[-(len(w)*2)//freq_filt_param:], w[:(len(w)*2)//freq_filt_param]]),
             np.concatenate([np.abs(A)[-(len(w)*2)//freq_filt_param:],
                            np.abs(A)[:(len(w)*2)//freq_filt_param]]),
             color='red')
    plt.show()
