import numpy as np
from scipy import fft
from matplotlib import pyplot as plt

t = np.arange(1000)
a = 10 + .2*np.sin(t/50)\
    + .08*np.sin(t/8+3) + .06*np.sin(t/7-4) + .033*np.sin(t/6+6) + .02*np.sin(t/5+8) \
    + .08*np.sin(t/4+3) + .06*np.sin(t/3.3-4) + .033*np.sin(t/3+6) + .02*np.sin(t/2.6+8)


longer_a = np.concatenate((a[::-1][-100:], a, a[::-1][:100]))
longer_t = np.arange(len(longer_a))-100
l = len(longer_t)

A = fft.fft(longer_a, norm='forward')
w = fft.fftfreq(len(longer_t))

half_win_width = l//200+1
window = np.zeros(l, dtype=float)
window[:half_win_width] = 1.0
window[len(window)-(half_win_width-1):] = 1.0
A_filt = A*window

a_filt = np.real(fft.ifft(A_filt, norm='forward'))[100:-100]

fig, (ax1, ax2) = plt.subplots(2, 1, height_ratios=[2, 1])
ax1.plot(longer_t, longer_a, color='lightblue')
ax1.plot(t, a, color='blue')
ax1.plot(t, a_filt, color='black')
ax2.plot(w, np.abs(A), color='red')
plt.show()