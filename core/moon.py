"""
Moon position (alt/az) and illumination fraction for a given moment,
used by the Tonight's Best moon-distance scoring criterion. Pure
astronomy - no Flask, no scoring weights or thresholds live here.
"""
from astropy.coordinates import get_body, AltAz, SkyCoord
import astropy.units as u

from astroplan.moon import moon_illumination


class MoonState:
    """Moon altitude/azimuth and illumination fraction at one instant."""

    __slots__ = ("altitude_deg", "azimuth_deg", "illumination", "_coord")

    def __init__(self, altitude_deg, azimuth_deg, illumination, coord):
        self.altitude_deg = altitude_deg
        self.azimuth_deg = azimuth_deg
        self.illumination = illumination
        self._coord = coord

    @property
    def is_up(self):
        """Whether the Moon is above the local horizon (0 deg)."""
        return self.altitude_deg > 0

    def separation_deg(self, ra_deg, dec_deg):
        """Angular separation, in degrees, between the Moon and a target."""
        target = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
        return self._coord.separation(target).deg


def moon_state_at(astropy_time, location, altaz=None):
    """
    Compute the Moon's alt/az and illumination fraction (0-1) at the
    given astropy Time/EarthLocation. An existing AltAz frame can be
    passed in to reuse one already built for the object calculations
    (see core.astro_calc.calc_time_location_and_lst).
    """
    if altaz is None:
        altaz = AltAz(obstime=astropy_time, location=location)

    moon_coord = get_body("moon", astropy_time, location)
    moon_altaz = moon_coord.transform_to(altaz)
    illumination = moon_illumination(astropy_time)

    return MoonState(
        altitude_deg=moon_altaz.alt.deg,
        azimuth_deg=moon_altaz.az.deg,
        illumination=float(illumination),
        coord=moon_coord,
    )
