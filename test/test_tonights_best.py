"""
Tests for the Tonight's Best modules: core.horizon, core.equipment,
core.moon, core.scoring. Run standalone with `python test/test_tonights_best.py`
or import run() from test/test.py style runners.
"""
import logging
import sys
import os
from datetime import datetime, timedelta

import pytz
from astropy.utils.iers import conf as iers_conf

# The sandbox this test runs in has no route to the IERS data servers.
# In a normally-networked deployment this isn't needed (astropy fetches
# the latest Earth-orientation table itself); set defensively so the
# test suite doesn't depend on outbound network access.
iers_conf.auto_max_age = None

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.horizon import HorizonProfile
from core.equipment import EquipmentSettings
from core.moon import moon_state_at
from core.scoring import (
    parse_size_arcmin,
    score_size_fit,
    score_brightness,
    score_moon_distance,
    ScoringContext,
)
from astropy.time import Time
from astropy.coordinates import EarthLocation
import astropy.units as u

logging.basicConfig(level=logging.WARNING)

passed_count = 0
failed_count = 0


def check(label, condition):
    global passed_count, failed_count
    if condition:
        print(f"PASS: {label}")
        passed_count += 1
    else:
        print(f"FAIL: {label}")
        failed_count += 1


# --- core.horizon ---------------------------------------------------

def test_horizon():
    flat = HorizonProfile.flat(5.0)
    check("flat horizon returns constant altitude", flat.min_altitude_at(0) == 5.0)
    check("flat horizon constant at other azimuth", flat.min_altitude_at(273.4) == 5.0)

    hrz_text = """
    4
    0 10
    90 20
    180 10
    270 20
    """
    profile = HorizonProfile.from_hrz_text(hrz_text)
    check("hrz parses exact point", profile.min_altitude_at(0) == 10.0)
    check("hrz interpolates midpoint", abs(profile.min_altitude_at(45) - 15.0) < 1e-9)
    check(
        "hrz wraps across 360/0 boundary",
        abs(profile.min_altitude_at(315) - 15.0) < 1e-9,
    )

    single = HorizonProfile([(90.0, 12.0)])
    check("single-point horizon returns that altitude everywhere", single.min_altitude_at(200) == 12.0)

    try:
        HorizonProfile([])
        check("empty points list raises", False)
    except ValueError:
        check("empty points list raises", True)


# --- core.equipment ---------------------------------------------------

def test_equipment():
    eq = EquipmentSettings(
        focal_length_mm=800,
        aperture_mm=200,
        sensor_width_mm=23.5,
        sensor_height_mm=15.6,
    )
    check("focal_ratio computed correctly", abs(eq.focal_ratio - 4.0) < 1e-9)
    # FOV = sensor_dim / focal_length (radians) -> arcmin
    expected_width_arcmin = (23.5 / 800) * (206264.80625 / 60)
    check(
        "fov_width_arcmin matches expected formula",
        abs(eq.fov_width_arcmin - expected_width_arcmin) < 1e-6,
    )

    eq_px = EquipmentSettings(
        focal_length_mm=800,
        aperture_mm=200,
        sensor_width_px=6248,
        sensor_height_px=4176,
        pixel_size_um=3.76,
    )
    check(
        "pixel-based sensor width matches mm equivalent",
        abs(eq_px.sensor_width_mm - (6248 * 3.76 / 1000)) < 1e-9,
    )

    try:
        EquipmentSettings(focal_length_mm=800, aperture_mm=200)
        check("missing sensor dims raises", False)
    except ValueError:
        check("missing sensor dims raises", True)

    try:
        EquipmentSettings(focal_length_mm=-1, aperture_mm=200, sensor_width_mm=1, sensor_height_mm=1)
        check("negative focal length raises", False)
    except ValueError:
        check("negative focal length raises", True)

    from_dict = EquipmentSettings.from_request_dict(
        {
            "focal_length_mm": 800,
            "aperture_mm": 200,
            "sensor_width_mm": 23.5,
            "sensor_height_mm": 15.6,
        }
    )
    check("from_request_dict builds equivalent settings", from_dict.sensor_width_mm == 23.5)


# --- core.moon ---------------------------------------------------

def test_moon():
    t = Time("2026-07-06 12:00:00")
    location = EarthLocation(lat=-33.7 * u.deg, lon=151.2 * u.deg, height=0 * u.m)
    state = moon_state_at(t, location)
    check("moon altitude is a plausible degree value", -90 <= state.altitude_deg <= 90)
    check("moon azimuth is within 0-360", 0 <= state.azimuth_deg < 360)
    check("moon illumination is a fraction", 0.0 <= state.illumination <= 1.0)

    sep_same_place = state.separation_deg(
        ra_deg=(state.azimuth_deg * 0) + 0, dec_deg=0
    )  # arbitrary target, just checking it runs
    check("separation_deg returns a non-negative number", sep_same_place >= 0)


# --- core.scoring ---------------------------------------------------

def test_scoring_helpers():
    check("size parse single value -> circular", parse_size_arcmin("12.9") == (12.9, 12.9))
    check("size parse WxH", parse_size_arcmin("6.0x4.0") == (6.0, 4.0))
    check("size parse WxH takes major/minor regardless of order", parse_size_arcmin("4.0x6.0") == (6.0, 4.0))
    check("size parse empty -> None", parse_size_arcmin("") is None)
    check("size parse None -> None", parse_size_arcmin(None) is None)
    check("size parse garbage -> None", parse_size_arcmin("abc") is None)

    # Object filling ~60% of a 60 arcmin frame -> right in the sweet spot
    fov = 60.0
    check(
        "size fit peaks in the 40-80% sweet spot",
        score_size_fit("36", fov, fov) == 1.0,
    )
    check(
        "size fit drops for a tiny object",
        score_size_fit("1", fov, fov) < 0.5,
    )
    check(
        "size fit penalises under-fill quadratically, not linearly",
        # NGC 6744 (15.5') in a Skywatcher Quattro 150P + ASI2600 FOV
        # (~107.7 x 72.0 arcmin, limiting dimension 72.0): fill fraction
        # ~21.5%, well under the 40% sweet-spot floor. A linear ramp
        # scored this ~0.54 (too generous for an object that's
        # genuinely small and lost in the frame); squared, it should
        # land well under a third.
        abs(score_size_fit("15.5", 107.7, 72.0) - 0.290) < 0.01,
    )
    check(
        "size fit is zero when object can't fit at all",
        score_size_fit("120", fov, fov) == 0.0,
    )
    check(
        "size fit unknown size assumes small framing fit",
        score_size_fit("", fov, fov) == 0.05,
    )
    check(
        "size fit unknown nebula size gets a less severe fallback",
        score_size_fit("", fov, fov, object_type="Nebula") == 0.35,
    )
    check(
        "size fit unknown galaxy size remains strongly penalised",
        score_size_fit("", fov, fov, object_type="Galaxy") == 0.05,
    )

    check("brightness score maxes out for very bright objects", score_brightness("2.0") == 1.0)
    check("brightness score bottoms out for very faint objects", score_brightness("16.0") == 0.0)
    check("brightness score midrange is between 0 and 1", 0 < score_brightness("9.0") < 1)
    check("brightness score unknown is neutral", score_brightness("") == 0.5)
    check("brightness score unknown (None) is neutral", score_brightness(None) == 0.5)

    class FakeMoon:
        def __init__(self, is_up, illumination):
            self.is_up = is_up
            self.illumination = illumination

    check(
        "moon below horizon never penalises",
        score_moon_distance(0.0, FakeMoon(is_up=False, illumination=1.0)) == 1.0,
    )
    check(
        "new moon barely penalises even close by",
        score_moon_distance(1.0, FakeMoon(is_up=True, illumination=0.0)) == 1.0,
    )
    check(
        "full moon right on target scores worst",
        score_moon_distance(0.0, FakeMoon(is_up=True, illumination=1.0)) == 0.0,
    )
    check(
        "full moon far away scores well",
        score_moon_distance(180.0, FakeMoon(is_up=True, illumination=1.0)) == 1.0,
    )


def test_scoring_context_end_to_end():
    """Sanity check the full ScoringContext pipeline runs and ranks sensibly."""
    equipment = EquipmentSettings(
        focal_length_mm=800, aperture_mm=200, sensor_width_mm=23.5, sensor_height_mm=15.6
    )
    horizon = HorizonProfile.flat(0.0)
    location = EarthLocation(lat=-33.7 * u.deg, lon=151.2 * u.deg, height=0 * u.m)

    timezone = pytz.timezone("Australia/Sydney")
    dusk = timezone.localize(datetime(2026, 7, 6, 18, 0, 0))
    dawn = dusk + timedelta(hours=8)

    context = ScoringContext(equipment=equipment, horizon=horizon, location=location, dusk=dusk, dawn=dawn)

    # A circumpolar-ish southern target (should be up for a healthy run
    # from a Sydney-latitude site) vs. a target that's below the horizon
    # for the whole window (large positive Dec from a southern site).
    good_target = context.score_object(ra_deg=180.0, dec_deg=-60.0, magnitude="6.0", size_str="20")
    poor_target = context.score_object(ra_deg=180.0, dec_deg=80.0, magnitude="6.0", size_str="20")

    check("good target has a positive shooting run", good_target.run_minutes > 0)
    check("target far below horizon has no shooting run", poor_target.run_minutes == 0)
    check("good target scores higher overall", good_target.composite > poor_target.composite)
    check(
        "component scores are all within [0, 1.01]",
        all(0 <= s <= 1.01 for s in [good_target.size_score, good_target.brightness_score, good_target.moon_score]),
    )


def test_size_gates_composite_score():
    """
    Size fit is a multiplicative GATE on the composite, not just one
    more additive component: a bright, long-visible object that's a
    "speck in the frame" should score decisively lower than a dimmer,
    shorter-visible object that actually fills the frame well -
    matching the real-world judgement that a poor framing fit isn't
    worth shooting no matter how good everything else is.
    """
    # A wide-field rig (Quattro 150P + ASI2600-like FOV) where a 15.5'
    # galaxy is a poor (~21%) fill, same shape as the real NGC 6744 case.
    equipment = EquipmentSettings(
        focal_length_mm=750, aperture_mm=150, sensor_width_mm=23.5, sensor_height_mm=15.7
    )
    horizon = HorizonProfile.flat(25.0)
    location = EarthLocation(lat=-33.7 * u.deg, lon=151.2 * u.deg, height=0 * u.m)
    timezone = pytz.timezone("Australia/Sydney")
    dusk = timezone.localize(datetime(2026, 7, 8, 18, 30, 0))
    dawn = timezone.localize(datetime(2026, 7, 9, 5, 30, 0))
    context = ScoringContext(equipment=equipment, horizon=horizon, location=location, dusk=dusk, dawn=dawn)

    # Bright, up-all-night, but only a 15.5' object in a ~72'-limiting-dimension frame.
    poorly_framed_but_otherwise_great = context.score_object(
        ra_deg=180.0, dec_deg=-70.0, magnitude="6.0", size_str="15.5"
    )
    # Fainter and only briefly visible, but a near-ideal 40' fill of the same frame.
    well_framed_but_otherwise_mediocre = context.score_object(
        ra_deg=90.0, dec_deg=10.0, magnitude="10.5", size_str="40"
    )

    check(
        "a poor framing fit scores decisively lower than a good fit, despite worse brightness/visibility",
        well_framed_but_otherwise_mediocre.composite > poorly_framed_but_otherwise_great.composite,
    )
    check(
        "a badly-framed object's composite stays low even with excellent brightness and full-night visibility",
        poorly_framed_but_otherwise_great.composite < 0.3,
    )


# --- flask_app.routes.objects (inline horizon resolution) ---------------------------------------------------

def test_inline_horizon_resolution():
    from flask_app.routes.objects import _resolve_horizon

    # No horizon_points supplied -> falls back to the server default
    # (flat DEFAULT_HORIZON_ALTITUDE_DEG, since no horizon.hrz is deployed
    # in this test environment).
    default_horizon = _resolve_horizon({})
    check(
        "missing horizon_points falls back to the flat server default",
        default_horizon.min_altitude_at(123.0) == 25.0,
    )

    # Client-supplied points (as sent by the browser after parsing an
    # imported .hrz landscape file) build a real polygon horizon.
    custom_horizon = _resolve_horizon({"horizon_points": [[0, 10], [90, 40], [180, 10], [270, 40]]})
    check("inline horizon_points builds a working polygon", custom_horizon.min_altitude_at(0) == 10.0)
    check("inline horizon_points interpolates correctly", abs(custom_horizon.min_altitude_at(45) - 25.0) < 1e-9)

    try:
        _resolve_horizon({"horizon_points": [["not-a-number", 10]]})
        check("malformed horizon_points raises", False)
    except ValueError:
        check("malformed horizon_points raises", True)

    # Sanity check against the real uploaded landscape file's shape:
    # a horizon with points well above the flat 25 deg default should
    # produce a *shorter* shooting-time run than the default fallback
    # for the same target, not a longer one.
    real_hrz_points = [
        (4, 41.7), (14, 39.7), (17, 33.7), (20, 36.9), (40, 38.8), (55, 39.2),
        (71, 35.7), (77, 35.4), (83, 37.6), (101, 34.9), (103, 29.8), (112, 28.7),
        (123, 29.3), (140, 33.4), (158, 31.6), (179, 29.9), (185, 26.4), (193, 31.2),
        (196, 30.2), (212, 27.8), (216, 19.8), (223, 18.6), (232, 19.2), (240, 26.1),
        (243, 41.8), (241, 50.4), (268, 58.3), (288, 57.8), (299, 56.3), (322, 50.2),
        (344, 45.3), (357, 41.1),
    ]
    real_horizon = _resolve_horizon({"horizon_points": [list(p) for p in real_hrz_points]})

    equipment = EquipmentSettings(
        focal_length_mm=800, aperture_mm=200, sensor_width_mm=23.5, sensor_height_mm=15.6
    )
    location = EarthLocation(lat=-33.87 * u.deg, lon=151.09 * u.deg, height=0 * u.m)
    timezone = pytz.timezone("Australia/Sydney")
    dusk = timezone.localize(datetime(2026, 4, 15, 18, 10, 0))
    dawn = timezone.localize(datetime(2026, 4, 16, 5, 30, 0))

    default_context = ScoringContext(
        equipment=equipment, horizon=default_horizon, location=location, dusk=dusk, dawn=dawn
    )
    real_context = ScoringContext(
        equipment=equipment, horizon=real_horizon, location=location, dusk=dusk, dawn=dawn
    )

    ra_deg = (16 + 31 / 60 + 30 / 3600) * 15
    dec_deg = -40.25
    default_result = default_context.score_object(ra_deg, dec_deg, magnitude="10.7", size_str="0.4")
    real_result = real_context.score_object(ra_deg, dec_deg, magnitude="10.7", size_str="0.4")

    check(
        "a real (higher) landscape horizon never gives a longer run than the flat default",
        real_result.run_minutes <= default_result.run_minutes,
    )


def run():
    test_horizon()
    test_equipment()
    test_moon()
    test_scoring_helpers()
    test_scoring_context_end_to_end()
    test_size_gates_composite_score()
    test_inline_horizon_resolution()

    total = passed_count + failed_count
    print("\n--- Summary ---")
    print(f"Total checks: {total}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")
    return failed_count == 0


if __name__ == "__main__":
    success = run()
    sys.exit(0 if success else 1)
