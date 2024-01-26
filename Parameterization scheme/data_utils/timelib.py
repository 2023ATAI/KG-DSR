import math
import datetime
import numpy as np

import xarray as xr
import pandas as pd
#from pytz import timezone
import pytz
from timezonefinder import TimezoneFinder
from datetime import datetime, timedelta

def get_timezone(latitude, longitude):
    tz = pytz.timezone('Etc/GMT')
    now = datetime.now(tz)
    offset_hours = round(longitude / 15)
    timezone_name = 'Etc/GMT{}'.format('+' if offset_hours < 0 else '-') + str(abs(offset_hours))
    timezone = pytz.timezone(timezone_name)
    timezone_name = timezone.tzname(now)
    if timezone.dst(now):
        timezone_name = timezone.tzname(now - timedelta(hours=1))
    return timezone_name


def get_local_etc_timezone(latitude, longitude):
    '''
    This function gets the time zone at a given latitude and longitude in 'Etc/GMT' format.
    This time zone format is used in order to avoid issues caused by Daylight Saving Time (DST) (i.e., redundant or
    missing times in regions that use DST).
    However, note that 'Etc/GMT' uses a counter intuitive sign convention, where West of GMT is POSITIVE, not negative.
    So, for example, the time zone for Zurich will be returned as 'Etc/GMT-1'.
    ...: 
     :param latitude: Latitude at the project location
     :param longitude: Longitude at the project location
    '''
    # get the time zone at the given coordinates
    tf = TimezoneFinder()
    time = pytz.timezone(tf.timezone_at(lng=longitude, lat=latitude)).localize(
         datetime(2011, 1, 1)).strftime('%z')
    # invert sign and return in 'Etc/GMT' format
    if time[0] == '-':
        time_zone = 'Etc/GMT+' + time[2]
    else:
        time_zone = 'Etc/GMT-' + time[2]
    return time[2]


def isleapyear(year):
    if ((year % 4 == 0) and (year % 100 != 0)) or (year % 400 == 0):
        return True
    else:
        return False


def calc_theta_s(xlat, xlong, stdlng, doy, year, ftime):
    """Calculates the Sun Zenith Angle (SZA).

    Parameters
    ----------
    xlat : float
        latitude of the site (degrees).
    xlong : float
        longitude of the site (degrees).
    stdlng : float
        central longitude of the time zone of the site (degrees).
    doy : float
        day of year of measurement (1-366).
    year : float
        year of measurement .
    ftime : float
        time of measurement (decimal hours).

    Returns
    -------
    theta_s : float
        Sun Zenith Angle (degrees).

    References
    ----------
    Adopted from Martha Anderson's fortran code for ALEXI which in turn was based on Cupid.
    """

    pid180 = np.pi / 180
    pid2 = np.pi / 2.0

    # Latitude computations
    xlat = np.radians(xlat)
    sinlat = np.sin(xlat)
    coslat = np.cos(xlat)

    # Declination computations
    kday = (year - 1977.0) * 365.0 + doy + 28123.0
    xm = np.radians(-1.0 + 0.9856 * kday)
    delnu = (2.0 * 0.01674 * np.sin(xm)
             + 1.25 * 0.01674 * 0.01674 * np.sin(2.0 * xm))
    slong = np.radians((-79.8280 + 0.9856479 * kday)) + delnu
    decmax = np.sin(np.radians(23.44))
    decl = np.arcsin(decmax * np.sin(slong))
    sindec = np.sin(decl)
    cosdec = np.cos(decl)
    eqtm = 9.4564 * np.sin(2.0 * slong) / cosdec - 4.0 * delnu / pid180
    eqtm = eqtm / 60.0

    # Get sun zenith angle
    timsun = ftime  # MODIS time is already solar time
    hrang = (timsun - 12.0) * pid2 / 6.0
    theta_s = np.arccos(sinlat * sindec + coslat * cosdec * np.cos(hrang))

    # if the sun is below the horizon just set it slightly above horizon
    theta_s = np.minimum(theta_s, pid2 - 0.0000001)
    theta_s = np.degrees(theta_s)

    return np.asarray(theta_s)


def calc_sun_angles(lat, lon, stdlon, doy, ftime):
    """Calculates the Sun Zenith and Azimuth Angles (SZA & SAA).

    Parameters
    ----------
    lat : float
        latitude of the site (degrees).
    long : float
        longitude of the site (degrees).
    stdlng : float
        central longitude of the time zone of the site (degrees).
    doy : float
        day of year of measurement (1-366).
    ftime : float
        time of measurement (decimal hours).

    Returns
    -------
    sza : float
        Sun Zenith Angle (degrees).
    saa : float
        Sun Azimuth Angle (degrees).
    """
    # Calculate declination
    declination = 0.409 * np.sin((2.0 * np.pi * doy / 365.0) - 1.39)
    EOT = (0.258 * np.cos(declination) - 7.416 * np.sin(declination)
           - 3.648 * np.cos(2.0 * declination) - 9.228 * np.sin(2.0 * declination))
    LC = (stdlon - lon) / 15.
    time_corr = (-EOT / 60.) + LC
    solar_time = ftime - time_corr

    # Get the hour angle
    w =(solar_time - 12.0) * 15.

    # Get solar elevation angle
    sin_thetha = (np.cos(np.radians(w)) * np.cos(declination) * np.cos(np.radians(lat))
                  + np.sin(declination) * np.sin(np.radians(lat)))
    sun_elev = np.arcsin(sin_thetha)

    # Get solar zenith angle
    sza = np.pi / 2.0 - sun_elev
    sza = np.degrees(sza)

    # Get solar azimuth angle
    cos_phi = (np.sin(declination) * np.cos(np.radians(lat))- np.cos(np.radians(w)) * np.cos(declination) * np.sin(np.radians(lat)))/ np.cos(sun_elev)
    saa = np.zeros(sza.shape)
    saa[w <= 0.0] = np.degrees(np.arccos(cos_phi[w <= 0.0]))
    saa[w > 0.0] = 360. - np.degrees(np.arccos(cos_phi[w > 0.0]))

    return np.asarray(sza) #, np.asarray(saa)



def orb_coszen(calday, dlon, dlat):
    # Constants
    dayspy = 365.0        # days per year
    ve = 80.5             # calday of vernal equinox assumes Jan 1 = calday 1
    eccen = 1.672393084E-2 # eccentricity
    obliqr = 0.409214646  # Earth's obliquity in radians
    lambm0 = -3.2625366E-2 # mean long of perihelion at the vernal equinox (radians)
    mvelpp = 4.92251015   # moving vernal equinox longitude of perihelion plus pi (radians)

    # Local variables
    pi = math.pi
    declin = 0.0
    eccf = 0.0
    lambm = lambm0 + (calday - ve) * 2.0 * pi / dayspy
    lmm = lambm - mvelpp
    sinl = math.sin(lmm)
    lamb = lambm + eccen * (2.0 * sinl + eccen * (1.25 * math.sin(2.0 * lmm) + eccen * ((13.0/12.0) * math.sin(3.0 * lmm) - 0.25 * sinl)))
    invrho = (1.0 + eccen * math.cos(lamb - mvelpp)) / (1.0 - eccen * eccen)
    declin = math.asin(math.sin(obliqr) * math.sin(lamb))
    eccf = invrho * invrho
    orb_coszen = math.sin(dlat) * math.sin(declin) - math.cos(dlat) * math.cos(declin) * math.cos(calday * 2.0 * pi + dlon)
    theta = np.degrees(np.arccos(orb_coszen)) 
    return orb_coszen,theta




