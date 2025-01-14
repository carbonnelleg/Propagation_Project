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

IridiumSatellites = []
# https://en.wikipedia.org/wiki/Iridium_satellite_constellation#Ground_stations
# https://www.n2yo.com/satellites/?c=15&srt=1&dir=1&p=0
for i in range(100, 190):
    try:
        IridiumSatellites.append(by_name[f'IRIDIUM {i}'])
    except KeyError as e:
        pass

GlobalStarSatellites = []
# https://en.wikipedia.org/wiki/Globalstar#Products_and_services
# https://www-sop.inria.fr/members/Eitan.Altman/DEASAT/deasat4.pdf
# https://space.skyrocket.de/doc_sdat/globalstar-2.htm
for i in range(73, 98):
    try:
        GlobalStarSatellites.append(by_name[f'GLOBALSTAR M{i:0>3d}'])
    except KeyError as e:
        pass


def relative_pos_evolution(
    satellite, ground_station, ndays=10, interval=10, dt=None
):

    # Reasonable error ranges for up to 10 days diff with epoch:
    # R = 2 km, I = 40 km, C = 1 km
    # --> range of 20 days centered around last observation day
    # TLE is not very precise but the error is very small for short period
    # RIC errors statistics for LEO satellites:
    # http://celestrak.org/publications/AAS/07-127/
    # RIC coordinate system
    # https://ai-solutions.com/_help_Files/attitude_reference_frames.htm
    relative_position = satellite - ground_station
    if dt is None:
        dt = satellite.epoch.utc_datetime()
    Y = dt.year
    M = dt.month
    D = dt.day
    times = [ts.utc(Y, M, D + d, 0, m, 0)
             for d in range(-ndays, ndays+1)
             for m in range(0, 24*60, interval)]
    altitudes = np.full_like(times, 0.0, dtype=float)
    azimuths = np.full_like(times, 0.0, dtype=float)
    distances = np.full_like(times, 0.0, dtype=float)
    for i, t in enumerate(times):
        topocentric = relative_position.at(t)
        # satellite position relative to ground station
        alt, az, dist = topocentric.altaz()
        altitudes[i] = alt.degrees
        azimuths[i] = az.degrees
        distances[i] = dist.km
    return (altitudes, azimuths, distances)


def calc_elev_metric(satellite, ground_station):
    print(f'### Calculationg elevation metric of {satellite.name} ###')
    altitudes, * \
        _ = relative_pos_evolution(
            satellite, ground_station, ndays=1, interval=1)
    count, bins = np.histogram(altitudes, bins=np.arange(-90., 90., 10.),
                               density=True)
    l = len(count)//2
    return 1/2*np.dot(count[l:], bins[l:-1] + bins[l+1:])


def plot_elev_angle(satellite, ground_station):
    print(f'### Plotting {satellite.name} ###')
    isLEO = wgs84.height_of(satellite.at(satellite.epoch)).km <= 2_000
    altitudes, azimuths, _ = relative_pos_evolution(
        satellite, ground_station, ndays=1, interval=1)
    dt = satellite.epoch.utc_datetime()

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


def best_elev_angle(relative_positions, t):

    i_max = 0
    alt_max = -90.0
    az_max = 0.0
    for i, pos in enumerate(relative_positions):
        topocentric = pos.at(t)
        alt, az, _ = topocentric.altaz()
        if alt.degrees > alt_max:
            i_max = i
            alt_max = alt.degrees
            az_max = az.degrees
    return (i_max, alt_max, az_max)


def plot_best_elev_angle(satellites, ground_station, name):
    relative_positions = [sat - ground_station for sat in satellites]
    dt = satellites[0].epoch.utc_datetime()
    Y = dt.year
    M = dt.month
    D = dt.day
    times = [ts.utc(Y, M, D + d, 0, m, 0)
             for d in range(-10, 11)
             for m in range(0, 24*60, 10)]
    altitudes = np.full_like(times, 0.0, dtype=float)
    azimuths = np.full_like(times, 0.0, dtype=float)
    for j, t in enumerate(times):
        i_max, alt_max, az = best_elev_angle(relative_positions, t)
        altitudes[j] = alt_max
        azimuths[j] = az
        print(i_max)

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(9, 7))
    ax1.scatter(azimuths, altitudes, s=.5)
    ax1.set_xlabel('Azimuth angle [°]')
    ax1.set_ylabel('Elevation angle [°]')
    ax1.grid(True)
    ax2.hist(altitudes, bins=np.arange(-90., 90., 10.))
    ax2.set_xlim((-95., 95.))
    ax2.set_xlabel('Elevation angle [°]')
    ax1.set_title(dt.date())
    fig.suptitle(name)
    fig.savefig(
        __file__ + f'/../figures/Satellites angles (Houston)/{name}.png')
    plt.close()


for sat in IridiumSatellites + GlobalStarSatellites:
    # uncomment to generate plots of elevation angles
    # plot_elev_angle(sat, Houston)
    # print(calc_elev_metric(sat, Houston))
    pass

plot_best_elev_angle(GlobalStarSatellites, Houston, 'GlobalStar')
plot_best_elev_angle(IridiumSatellites, Houston, 'Iridium')
