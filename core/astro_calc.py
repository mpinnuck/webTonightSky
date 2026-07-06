"""
Pure astronomy calculations: LST, transit time, alt/az, transit
altitude, and RA/Dec formatting helpers. No Flask or HTTP concerns
live here.
"""
from datetime import timedelta

import pytz
import astropy.units as u
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy.time import Time


def ra_to_degrees(ra_str):
    """Convert RA from 'HH:MM:SS' string to decimal degrees."""
    hours, minutes, seconds = map(float, ra_str.split(":"))
    return (hours + minutes / 60 + seconds / 3600) * 15


def degrees_to_ra(degrees):
    """Convert RA in decimal degrees to an 'HH:MM:SS' string."""
    hours = int(degrees // 15)
    minutes = int((degrees % 15) * 4)
    seconds = (degrees % 15) * 240 - minutes * 60
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d}"


def format_dec(dec):
    """Format a declination value in degrees with a trailing degree sign."""
    return f"{dec:.2f}°"


def format_transit_time(transit_time_minutes):
    """Format a duration given in minutes as an 'HH:MM:SS' string."""
    time_to_transit_seconds = abs(transit_time_minutes * 60)
    hours = int(time_to_transit_seconds // 3600)
    minutes = int((time_to_transit_seconds % 3600) // 60)
    seconds = int(time_to_transit_seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def calc_transit_altitude(ra_deg, dec_deg, latitude, longitude):
    """
    Calculate the transit altitude of a celestial object based on its
    declination and observer's latitude.

    The transit altitude is the maximum altitude an object reaches when
    it crosses the observer's meridian:
        Transit Altitude = 90 deg - |Latitude - Declination|

    Parameters:
    - ra_deg (float): Right Ascension in degrees (not used in this calc).
    - dec_deg (float): Declination of the object in degrees.
    - latitude (float): Observer's latitude in degrees.
    - longitude (float): Observer's longitude in degrees (not used here).

    Returns:
    - float: The transit altitude in degrees.
    """
    if not (-90 <= dec_deg <= 90):
        raise ValueError("Declination must be between -90 and 90 degrees")
    if not (-90 <= latitude <= 90):
        raise ValueError("Latitude must be between -90 and 90 degrees")

    transit_alt = 90 - abs(latitude - dec_deg)
    return max(-90, min(90, transit_alt))


def calc_time_location_and_lst(latitude, longitude, local_time):
    """Build the Astropy Time/EarthLocation/AltAz frame and LST for a moment."""
    astropy_time = Time(local_time.astimezone(pytz.utc))
    location = EarthLocation(lat=latitude * u.deg, lon=longitude * u.deg, height=0 * u.m)
    altaz = AltAz(obstime=astropy_time, location=location)
    lst = astropy_time.sidereal_time("mean", longitude * u.deg).hour
    return astropy_time, location, altaz, lst


def calc_transit_and_alt_az(ra_deg, dec_deg, local_time, astropy_time, location, altaz, lst):
    """Calculate transit time offset, alt/az, and direction for a target."""
    target = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
    altaz_coord = target.transform_to(altaz)
    altitude = altaz_coord.alt.deg
    azimuth = altaz_coord.az.deg

    ra_hours = ra_deg / 15.0
    time_diff_hours = ra_hours - lst
    if time_diff_hours > 12:
        time_diff_hours -= 24
    elif time_diff_hours < -12:
        time_diff_hours += 24

    before_after = "After" if time_diff_hours >= 0 else "Before"
    transit_time_minutes = abs(time_diff_hours * 60)
    local_transit_time = local_time + timedelta(
        minutes=transit_time_minutes if before_after == "After" else -transit_time_minutes
    )
    direction = "south" if 90 < azimuth < 270 else "north"

    return (
        transit_time_minutes,
        local_transit_time.strftime("%H:%M:%S"),
        before_after,
        altitude,
        azimuth,
        direction,
    )


def calculate_transit_and_alt_az(ra_deg, dec_deg, latitude, longitude, local_time):
    """Convenience wrapper combining calc_time_location_and_lst + calc_transit_and_alt_az."""
    astropy_time, location, altaz, lst = calc_time_location_and_lst(latitude, longitude, local_time)
    return calc_transit_and_alt_az(ra_deg, dec_deg, local_time, astropy_time, location, altaz, lst)
