# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 10:14:16 2025

@author: carbonnelleg

Interesting links:
    http://aprs.org/LEO-tracking.html
"""

import numpy as np
import matplotlib.pyplot as plt
from skyfield.api import load, wgs84, EarthSatellite

# Location of receiving Earth station
# Enter latitude [°], longitude [°] and altitude [m] of your ground station
LLN = wgs84.latlon(50.67, 4.61, 160)  # LLN for example
# NASA Johnson Space Center in Houston
Houston = wgs84.latlon(29.5594, -95.0903, 500)
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


# https://en.wikipedia.org/wiki/Iridium_satellite_constellation#Ground_stations
# https://www.n2yo.com/satellites/?c=15&srt=1&dir=1&p=0
# Iridium NEXT project; 80 satellites constellation
# height = 780
IridiumSatellites = []
for i in range(100, 181):
    try:
        IridiumSatellites.append(by_name[f'IRIDIUM {i:0>3d}'])
    except KeyError as e:
        pass

# https://en.wikipedia.org/wiki/Globalstar#Products_and_services
# https://www-sop.inria.fr/members/Eitan.Altman/DEASAT/deasat4.pdf
# https://space.skyrocket.de/doc_sdat/globalstar-2.htm
# GlobalStar 2nd generation; 25 satellites constellation
# height = 1414
GlobalStarSatellites = []
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
    neg_alt = np.full_like(times, False, dtype=bool)
    for i, t in enumerate(times):
        topocentric = relative_position.at(t)
        # satellite position relative to ground station
        alt, az, dist = topocentric.altaz()
        altitudes[i] = alt.degrees
        azimuths[i] = az.degrees
        distances[i] = dist.km
        neg_alt[i] = alt.degrees < 1
    # distance[i] should be related to altitudes[i] by the following equation
    # dist = -R*sin(alt) + sqrt( R**2*sin(alt)**2 + h**2 + 2*R*h )
    # where R is Earth's radius and h is height of satellites above R (constant)
    # This formula is yielded by applying Law of cosines in a triangle
    return (altitudes, azimuths, distances, neg_alt)


def calc_elev_metric(satellite, ground_station):
    print(f'### Calculating elevation metric of {satellite.name} ###')
    altitudes, _, _, neg_alt = relative_pos_evolution(
        satellite, ground_station, ndays=1, interval=1)
    if neg_alt.any():
        return altitudes[~neg_alt].sum()/len(altitudes)
    else:
        return altitudes.sum()/len(altitudes)


def plot_elev_angle(satellite, ground_station):
    print(f'### Plotting elevation of {satellite.name} ###')
    # Determine if satellite is LEO, but with certain probability of mistake
    # If the satellite follows elliptic curve and is near perigee, then result might be wrong
    isLEO = wgs84.height_of(satellite.at(satellite.epoch)).km <= 2_000
    # isLEO is not used here but could help discriminate LEO satellites from non-LEO
    altitudes, azimuths, distances, neg_alt = relative_pos_evolution(
        satellite, ground_station, ndays=1, interval=1)
    dt = satellite.epoch.utc_datetime()

    fig, (ax1, ax2) = plt.subplots(
        nrows=2, height_ratios=(3, 2), figsize=(9, 7))
    scatter = ax1.scatter(azimuths[~neg_alt], altitudes[~neg_alt], s=3.,
                          c=distances[~neg_alt], cmap='jet',
                          vmin=0., vmax=np.max(distances))
    if neg_alt.any():
        ax1.scatter(azimuths[neg_alt], altitudes[neg_alt], s=.4, c='gray')
    ax1.set_xlabel('Azimuth angle [°]')
    ax1.set_ylabel('Elevation angle [°]')
    ax1.grid(True)
    fig.colorbar(scatter, ax=ax1, orientation='vertical',
                 label='Distance [km]')

    bins = np.arange(-92.5, 97.5, 5.)
    bins[0] = -90.
    bins[-1] = 90.
    counts, *_ = ax2.hist(altitudes, bins=bins, color='green')
    ax2.set_xlim((-95., 95.))
    ax2.set_xlabel('Elevation angle [°]')
    ax1.set_title(dt.date())
    fig.suptitle(satellite.name)

    fig.savefig(
        __file__ + f'/../figures/Satellites angles (Houston)/{satellite.name.replace("/", "")}.png')
    plt.close()

    return (counts, bins)


def best_elev_angle(relative_positions, t):
    best_i = 0
    alt_max = -90.0
    best_az = 0.0
    best_dist = 0.0
    for i, pos in enumerate(relative_positions):
        topocentric = pos.at(t)
        alt, az, dist = topocentric.altaz()
        if alt.degrees > alt_max:
            best_i = i
            alt_max = alt.degrees
    _, best_az, best_dist = relative_positions[best_i].at(t).altaz()
    return (best_i, alt_max, best_az.degrees, best_dist.km)


def plot_best_elev_angle(satellites, ground_station, name):
    print(f'### Plotting elevation of {name} ###')
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
    distances = np.full_like(times, 0.0, dtype=float)
    for j, t in enumerate(times):
        _, alt_max, az, dist = best_elev_angle(relative_positions, t)
        altitudes[j] = alt_max
        azimuths[j] = az
        distances[j] = dist

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(9, 7))
    scatter = ax1.scatter(azimuths, altitudes, s=1.,
                          c=distances, cmap='jet',
                          vmin=0., vmax=np.max(distances))
    ax1.set_xlabel('Azimuth angle [°]')
    ax1.set_ylabel('Elevation angle [°]')
    ax1.grid(True)
    fig.colorbar(scatter, ax=ax1, orientation='vertical',
                 label='Distance [km]')

    bins = np.arange(-92.5, 97.5, 5.)
    bins[0] = -90.
    bins[-1] = 90.
    counts, *_ = ax2.hist(altitudes, bins=bins)
    ax2.set_xlim((-95., 95.))
    ax2.set_xlabel('Elevation angle [°]')
    ax1.set_title(dt.date())
    fig.suptitle(name)
    fig.savefig(
        __file__ + f'/../figures/Satellites angles (Houston)/{name}.png')
    plt.close()

    return (counts, bins)


def calc_best_elev_metrics(satellites, ground_station, name):
    print(f'### Calculating elevation metric of {name} ###')
    relative_positions = [sat - ground_station for sat in satellites]
    dt = satellites[0].epoch.utc_datetime()
    Y = dt.year
    M = dt.month
    D = dt.day
    times = [ts.utc(Y, M, D + d, 0, m, 0)
             for d in range(-1, 2)
             for m in range(0, 24*60, 1)]
    altitudes = np.full_like(times, 0.0, dtype=float)
    for j, t in enumerate(times):
        _, alt_max, *_ = best_elev_angle(relative_positions, t)
        altitudes[j] = alt_max
    return altitudes.mean()


for sat in IridiumSatellites + GlobalStarSatellites:
    # uncomment to generate plots of elevation angles
    # plot_elev_angle(sat, Houston)
    # print(calc_elev_metric(sat, Houston))
    pass

# plot_best_elev_angle(GlobalStarSatellites, Houston, 'GlobalStar')
# plot_best_elev_angle(IridiumSatellites, Houston, 'Iridium')
# print(calc_best_elev_metrics(IridiumSatellites, Houston, 'Iridium Satellites'))
# print(calc_best_elev_metrics(GlobalStarSatellites, Houston, 'GlobalStar Satellites'))
