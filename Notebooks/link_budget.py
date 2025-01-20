# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 19:32:50 2025

@author: carbonnelleg

Interesting links:
    https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.676-8-200910-S!!PDF-E.pdf
    https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.840-4-200910-S!!PDF-E.pdf
    https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.618-10-200910-S!!PDF-E.pdf
"""
import pandas as pd
import numpy as np
import utils
import satellite_elevation as se
from matplotlib import pyplot as plt

EPSILON = 1e-5

# Read files and generate pandas DataFrame
# field_names = \
# ['SITE', 'SATELLITE', 'FREQUENCY', 'ELEVATION', 'EP', 'PROBABILITY', 'ATTENUATION']
field_names = ['SITE', 'SATELLITE', 'FREQUENCY', 'ATTENUATION', 'PROBABILITY']
# depends on the type of file that is read! E.g. for total attenuation

cdf = True
file1 = pd.read_csv(
    __file__ + '/../../Houston/houston_37GHz_1/output/ascii/attenuation_total_cdf.csv',
    skiprows=7, names=field_names)
file2 = pd.read_csv(
    __file__ + '/../../Houston/houston_37GHz_2/output/ascii/attenuation_total_cdf.csv',
    skiprows=7, names=field_names)

rapids_data = pd.concat((file1, file2), ignore_index=True)
rapids_data.drop_duplicates(inplace=True, ignore_index=True)
if cdf:
    rapids_data['ELEVATION'] = (rapids_data.index//26+1)*5.

# utils.plot_RAPIDS_outputs(rapids_data, 'Total attenuation for LLN at 37.5 GHz')

satellites = se.IridiumSatellites
counts, bins = se.plot_best_elev_angle(
    se.IridiumSatellites, se.Houston, 'Iridium')
# counts, bins = se.plot_best_elev_angle(
#     se.GlobalStarSatellites, se.Houston, 'GlobalStar')
# counts, bins = se.plot_elev_angle(se.AlphaSat, se.LLN)
# counts, bins = se.plot_elev_angle(satellites[10], se.Houston)

s = counts[19:].sum()
link_budget_df = pd.DataFrame()
probs = np.zeros_like(rapids_data.index, dtype=float)

print('### Computing Link Budget ###')
for j, att in rapids_data.loc[:, 'ATTENUATION'].items():
    p = 0.0
    for i, (theta_low, theta_high) in enumerate(zip(bins[19:-1], bins[20:]), start=19):
        my_df = rapids_data.loc[
            (rapids_data['ELEVATION'] >= theta_low) &
            (rapids_data['ELEVATION'] <= theta_high)].loc[
                (rapids_data['ATTENUATION'] >= att)]
        if not my_df.empty:
            p += my_df.iloc[-1].loc['PROBABILITY']*counts[i]/s
    probs[j] = p

link_budget_df['ATTENUATION'] = rapids_data['ATTENUATION']
link_budget_df['PROBABILITY'] = probs
link_budget_df.sort_values(by='ATTENUATION', ascending=False,
                           inplace=True, ignore_index=True)

link_budget_smooth = pd.DataFrame()
link_budget_smooth['PROBABILITY'] = rapids_data.loc[:, 'PROBABILITY'].iloc[:26]
attenuations = np.zeros_like(link_budget_smooth.index, dtype=float)
for i, p in link_budget_smooth.loc[:, 'PROBABILITY'].items():
    attenuations[i] = link_budget_df.loc[
        (link_budget_df['PROBABILITY'] >= p-EPSILON), 'ATTENUATION'].iloc[0]
link_budget_smooth['ATTENUATION'] = attenuations

utils.plot_RAPIDS_outputs(link_budget_smooth, 'Test', link_budget=True)
plt.title(f'Above horizon {100*s/counts.sum():.2f} % of time')
