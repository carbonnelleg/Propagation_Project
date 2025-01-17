# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 19:32:50 2025

@author: carbonnelleg
"""
import pandas as pd
import numpy as np
import utils
import satellite_elevation as se

# Read files and generate pandas DataFrame
field_names = ['SITE', 'SATELLITE', 'FREQUENCY', 'ELEVATION', 'EP', 'PROBABILITY',
               'ATTENUATION']
# depends on the type of file that is read! E.g. for total attenuation

file1 = pd.read_csv(
    __file__ + '/../rapids/lln_37GHz_1/output/ascii/attenuation_total.csv',
    skiprows=7, names=field_names)
file2 = pd.read_csv(
    __file__ + '/../rapids/lln_37GHz_2/output/ascii/attenuation_total.csv',
    skiprows=7, names=field_names)

rapids_data = pd.concat((file1, file2), ignore_index=True)

# utils.plot_RAPIDS_outputs(rapids_data, "Total attenuation for LLN at 37.5 GHz")

satellites = se.GlobalStarSatellites
# counts, bins = se.plot_best_elev_angle(satellites, se.Houston, 'Iridium')
# counts, bins = se.plot_elev_angle(se.AlphaSat, se.LLN)
counts, bins = se.plot_elev_angle(satellites[0], se.Houston)

s = counts.sum()
link_budget_df = pd.DataFrame()
probs = np.zeros_like(rapids_data.index, dtype=float)

print('### Computing Link Budget ###')
for j, att in rapids_data.loc[:, 'ATTENUATION'].items():
    p = 0.0

    for i, (theta_low, theta_high) in enumerate(zip(bins[:-1], bins[1:])):
        if theta_low < 0.:
            p += 100*counts[i]/s
        else:
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
