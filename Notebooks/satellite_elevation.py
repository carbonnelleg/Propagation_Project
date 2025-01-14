# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 10:14:16 2025

@author: carbonnelleg
"""

import numpy as np
import matplotlib.pyplot as plt
from skyfield.api import load, wgs84, EarthSatellite

# Location of receiving Earth station
# Enter latitude [°], longitude [°] and altitude [m] of your ground station
LLN = wgs84.latlon(50.67, 4.61, 160)  # LLN for example
# NASA Johnson Space Center in Houston
Houston = wgs84.latlon(29.5594, -95.0903)
ts = load.timescale()

satellite_url = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle'
satellites = load.tle_file(satellite_url, reload=False)

by_name = {sat.name: sat for sat in satellites}
HST = by_name['HST']  # find Hubble Space Telescope satellite
print(HST)

line1 = '1 39215U 13038A   25013.68368907  .00000133  00000-0  00000-0 0  9990'
line2 = '2 39215   2.9165   5.6981 0002326 297.2202  81.4506  1.00268209 38813'
AlphaSat = EarthSatellite(line1, line2, 'AlphaSat', ts)
print(AlphaSat)

satellite = HST


def plot_elev_angle(satellite, ground_station):
    print(f'### Plotting {satellite.name} ###')
    relative_position = satellite - Houston

    isLEO = wgs84.height_of(satellite.at(satellite.epoch)).km <= 2_000
    dt = satellite.epoch.utc_datetime()
    y = dt.year
    m = dt.month
    D = dt.day
    # range of 20 days centered around last observation day
    # TLE is not very precise but the error is very small for short period
    # RIC errors statistics for LEO satellites:
    # http://celestrak.org/publications/AAS/07-127/
    # RIC coordinate system
    # https://ai-solutions.com/_help_Files/attitude_reference_frames.htm
    times = [ts.utc(y, m, d + D, H, M, 0) for d in range(-10, 11)
             for H in range(0, 24) for M in range(0, 60, 10)]

    altitudes = np.full_like(times, 0.0, dtype=float)
    azimuths = np.full_like(times, 0.0, dtype=float)
    for i, t in enumerate(times):
        # geocentric = satellite.at(t)
        # satellite position in Geocentric Celestial Reference System (GCRS)
        # print(geocentric.position.km)

        # satellite position in latitude, longitude
        # lat, lon = wgs84.latlon_of(geocentric)
        # print('Latitude:', lat.dstr(format=u'{0}{1}°{2:02}′{3:02}.{4:0{5}}″'))
        # print('Longitude:', lon.dstr(format=u'{0}{1}°{2:02}′{3:02}.{4:0{5}}″'))

        topocentric = relative_position.at(t)
        # satellite position relative to ground station
        alt, az, distance = topocentric.altaz()
        altitudes[i] = alt.degrees
        azimuths[i] = az.degrees
        # print('Altitude:', alt)
        # print('Azimuth:', az)
        # print(f'Distance: {distance.km:.1f} km')

    color = 'green' if isLEO else 'red'
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(9, 7))
    ax1.scatter(azimuths, altitudes, s=.5, c=color)
    ax1.set_xlabel('Azimuth angle [°]')
    ax1.set_ylabel('Elevation angle [°]')
    ax1.grid(True)
    ax2.hist(altitudes, bins=np.arange(-90., 90., 10.), color=color)
    ax2.set_xlim((-95., 95.))
    ax2.set_xlabel('Elevation angle [°]')
    ax1.set_title(dt.date())
    fig.suptitle(satellite.name)
    fig.savefig(
        __file__ + f'/../figures/Satellites angles (Houston)/{satellite.name.replace("/", "")}.png')
    plt.close()


for sat in satellites:
    plot_elev_angle(sat, Houston)
