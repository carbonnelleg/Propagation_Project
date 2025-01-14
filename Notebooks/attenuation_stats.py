# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 09:12:27 2025

@author: carbonnelleg
"""

import numpy as np
from matplotlib import pyplot as plt

# Generate random samples using numpy Weibull distribution
shape_param = .5
rand_attenuations = np.random.weibull(a=shape_param, size=(100_000,))
# [2.10826815 2.24550052 0.28790755 ... 0.05630824 0.25578806 0.01657083]
# pdf defined here: https://en.wikipedia.org/wiki/Weibull_distribution
# print(rand_attenuations)


def weib_pdf(x, n, a):
    return (a / n) * (x / n)**(a - 1) * np.exp(-(x / n)**a)


def weib_cdf(x, n, a):
    return 1 - np.exp(-(x / n)**a)


# vertical histogram, width of bins is bin_width
bin_width = 2.
fig, ax = plt.subplots()
ax.hist(rand_attenuations,
        bins=np.arange(0., rand_attenuations.max(), bin_width))
fig.suptitle('Vertical Histogram')

# horizontal, cumulative histogram
fig, ax = plt.subplots()
ax.hist(rand_attenuations, orientation='horizontal', density=True, cumulative=-1,
        bins=np.arange(0., rand_attenuations.max(), bin_width))
fig.suptitle('Horizontal Cumulative Histogram')

# typical CCDF graph
fig, ax = plt.subplots()
cum_count, bins, _ = ax.hist(rand_attenuations, orientation='horizontal',
                             density=True, cumulative=-1, histtype='stepfilled',
                             color='lightblue', bins=np.arange(
                                 0., rand_attenuations.max(), bin_width))
count, bins = np.histogram(rand_attenuations, density=True,
                           bins=np.arange(0., rand_attenuations.max(), bin_width))
cum_count = count[::-1].cumsum()[::-1]*bin_width
attenuations = bins[:-1]
ax.plot(cum_count, attenuations+bin_width/2, label='Simulation')
weib_th = 1 - weib_cdf(attenuations, 1., shape_param)
ax.plot(weib_th, attenuations, label='Theory')
ax.set_xscale('log')
ax.set_xticks(ax.get_xticks(), labels=[
              f'{i*100:.0e}' for i in ax.get_xticks()])
ax.set_xlim((1e-6, 1e0))
ax.set_xlabel('Percentage [%]')
ax.set_ylabel('Attenuation [dB]')
ax.legend()
ax.set_title(f'Weibull Distribution (shape parameter = {shape_param:.1f})')
fig.suptitle('CCDF')

plt.show()
