"""
Tonight's Best composite scoring: combines size-fit, brightness,
shooting-time, and moon-distance into a single ranking score used to
re-order an already-filtered list_objects result set. This module owns
the scoring math only; it doesn't touch the query language, the
catalog, or HTTP concerns.
"""
import re
from collections import namedtuple
from datetime import timedelta

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import AltAz, SkyCoord

from core.config import logger
from core.moon import moon_state_at

# Size fit is a multiplicative GATE on the composite score, not one
# more additive component: a poor framing fit should crush the overall
# score regardless of how bright or long-visible the object is (a
# "speck in the frame" isn't worth shooting no matter what), while a
# good fit leaves brightness/time/moon to rank normally among
# similarly well-framed candidates, even if that means a short
# shooting window. Composite = size_score * (weighted rest).
# These weight the *rest* (brightness > shooting time > moon distance,
# the priority order agreed for Tonight's Best) and sum to 1.0; they
# are a tunable starting point, not a derived result.
BRIGHTNESS_WEIGHT = 0.5
TIME_WEIGHT = 1.0 / 3.0
MOON_WEIGHT = 1.0 / 6.0

# Brightness normalisation range (magnitude). Objects at or brighter
# than MAG_BRIGHT score 1.0; at or fainter than MAG_FAINT score 0.0.
# Fixed scale rather than relative-to-result-set, so scores are
# comparable across different filter queries.
MAG_BRIGHT = 4.0
MAG_FAINT = 14.0

# Size-fit target: the fraction of the frame's limiting dimension the
# object's major axis should fill for a "well framed" score of 1.0.
SIZE_FILL_LOW = 0.4
SIZE_FILL_HIGH = 0.8

# Base score when an object's size is missing from the catalog.
# Type-specific fallbacks can override this in unknown_size_score().
SIZE_UNKNOWN_SCORE = 0.05

# Sampling interval across the night when hunting for the longest
# continuous clear run. The horizon polygon isn't analytically
# invertible, so this is a scan rather than a closed-form solve.
# 10 minutes keeps per-request astropy transforms modest; if short
# visibility windows near the horizon prove undercounted in practice,
# reduce to 5 minutes at roughly 2x coordinate-transform cost.
SAMPLE_INTERVAL_MINUTES = 10

_SIZE_RE = re.compile(r"^\s*([\d.]+)\s*[xX]\s*([\d.]+)\s*$")

ScoreBreakdown = namedtuple(
    "ScoreBreakdown",
    [
        "composite",
        "size_score",
        "brightness_score",
        "time_score",
        "moon_score",
        "run_minutes",
        "peak_clearance_deg",
    ],
)


def parse_size_arcmin(size_str):
    """
    Parse the catalog Size field into (major_axis, minor_axis) arcmin.
    Accepts a single number ("12.9" -> circular, 12.9 x 12.9), a
    "WxH" pair ("6.0x4.0" -> 6.0 x 4.0), or an empty/unparsable value
    (returns None, meaning "unknown size").
    """
    if size_str is None:
        return None
    text = str(size_str).strip()
    if not text:
        return None

    match = _SIZE_RE.match(text)
    if match:
        a, b = float(match.group(1)), float(match.group(2))
        return (max(a, b), min(a, b))

    try:
        value = float(text)
        return (value, value)
    except ValueError:
        logger.warning(f"Unparsable Size value: {size_str!r}")
        return None


def unknown_size_score(object_type):
    """
    Type-aware fallback when size is missing.
    Nebulae are often under-catalogued/partial in "Size" fields, so
    they get a less severe penalty than galaxies.
    """
    if object_type is None:
        return SIZE_UNKNOWN_SCORE

    obj_type = str(object_type).strip().lower()
    if not obj_type:
        return SIZE_UNKNOWN_SCORE

    if "diffuse nebula" in obj_type or "emission nebula" in obj_type or "nebula" in obj_type:
        return 0.35
    if "cluster" in obj_type:
        return 0.20
    if "planetary nebula" in obj_type:
        return 0.10
    if "galaxy" in obj_type:
        return 0.05
    return SIZE_UNKNOWN_SCORE


def score_size_fit(size_str, fov_width_arcmin, fov_height_arcmin, object_type=None):
    """
    Score how well the object fills the frame, 0-1. Peaks (1.0) when
    the object's major axis fills between SIZE_FILL_LOW and
    SIZE_FILL_HIGH of the frame's limiting (smaller) dimension, so the
    fit assessment doesn't depend on framing orientation. Falls off on
    both sides; scores 0 if the object can't fit inside the frame at
    all in any orientation.

    Unknown size returns a type-aware fallback score so genuinely good
    nebula targets with missing catalog size are not buried, while
    unknown-size galaxies remain strongly penalised.

    The under-fill side falls off with the *square* of the fill
    fraction, not linearly: an object at half the target linear size
    only covers a quarter of the frame's area, and a small target
    lost in a wide field is a genuinely poor pick, not a middling one
    - a linear ramp scored these too generously (e.g. an object
    filling ~22% of the frame still scored ~0.54).
    """
    parsed = parse_size_arcmin(size_str)
    if parsed is None:
        return unknown_size_score(object_type)

    major, _minor = parsed
    limiting_dimension = min(fov_width_arcmin, fov_height_arcmin)
    if limiting_dimension <= 0 or major <= 0:
        return unknown_size_score(object_type)

    fill_fraction = major / limiting_dimension

    if fill_fraction > 1.0:
        return 0.0  # doesn't fit in the frame at all, in any orientation
    if fill_fraction < SIZE_FILL_LOW:
        return (fill_fraction / SIZE_FILL_LOW) ** 2
    if fill_fraction <= SIZE_FILL_HIGH:
        return 1.0
    return 1.0 - (fill_fraction - SIZE_FILL_HIGH) / (1.0 - SIZE_FILL_HIGH)


def score_brightness(magnitude):
    """
    Brighter (lower magnitude) scores higher, 0-1, clamped to the
    MAG_BRIGHT..MAG_FAINT range. Unknown/unparsable magnitude scores a
    neutral 0.5.
    """
    try:
        magnitude = float(magnitude)
    except (TypeError, ValueError):
        return 0.5

    if magnitude <= MAG_BRIGHT:
        return 1.0
    if magnitude >= MAG_FAINT:
        return 0.0
    return (MAG_FAINT - magnitude) / (MAG_FAINT - MAG_BRIGHT)


def score_moon_distance(separation_deg, moon_state):
    """
    1.0 when the Moon doesn't matter (below the horizon, or new/thin),
    down to 0.0 when a bright Moon sits right on top of the target.
    """
    if not moon_state.is_up:
        return 1.0

    separation_fraction = max(0.0, min(1.0, separation_deg / 180.0))
    penalty = moon_state.illumination * (1.0 - separation_fraction)
    return 1.0 - penalty


def build_sample_times(dusk, dawn, sample_minutes=SAMPLE_INTERVAL_MINUTES):
    """Build the shared list of sample datetimes spanning tonight's darkness."""
    times = []
    t = dusk
    while t <= dawn:
        times.append(t)
        t += timedelta(minutes=sample_minutes)
    # Avoid duplicate endpoint if the loop landed exactly on dawn.
    # Equality is reliable for this path because dusk/dawn come from the
    # same source and are expected to be minute-aligned datetimes.
    if times[-1] != dawn:
        times.append(dawn)
    return times


def find_best_shooting_window(ra_deg, dec_deg, location, sample_times, sample_time_array, horizon):
    """
    Scan the darkness window and find the longest continuous run where
    the object is above the local horizon
    (altitude(t) > horizon.min_altitude_at(azimuth(t))).

    sample_times: list of plain datetimes (for indexing/labelling).
    sample_time_array: an astropy Time built from the same instants,
    shared across objects so it's only constructed once per request.

    Returns a ScoreBreakdown-compatible tuple:
    (run_minutes, peak_clearance_deg, peak_time) for the best run
    found, or (0, 0.0, None) if the object is never clear.
    """
    target = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
    frame = AltAz(obstime=sample_time_array, location=location)
    altaz_coord = target.transform_to(frame)
    altitudes = altaz_coord.alt.deg
    azimuths = altaz_coord.az.deg

    best_run = 0
    best_peak_clearance = 0.0
    best_peak_time = None

    current_run = 0
    current_peak_clearance = 0.0
    current_peak_time = None

    def close_run():
        nonlocal best_run, best_peak_clearance, best_peak_time
        if current_run > best_run or (
            current_run == best_run and current_peak_clearance > best_peak_clearance
        ):
            best_run = current_run
            best_peak_clearance = current_peak_clearance
            best_peak_time = current_peak_time

    for i in range(len(sample_times)):
        clearance = altitudes[i] - horizon.min_altitude_at(azimuths[i])
        if clearance > 0:
            current_run += 1
            if clearance > current_peak_clearance:
                current_peak_clearance = clearance
                current_peak_time = sample_times[i]
        else:
            close_run()
            current_run = 0
            current_peak_clearance = 0.0
            current_peak_time = None

    close_run()

    interval_minutes = SAMPLE_INTERVAL_MINUTES
    if len(sample_times) > 1:
        interval_minutes = (sample_times[1] - sample_times[0]).total_seconds() / 60.0

    run_minutes = best_run * interval_minutes
    return run_minutes, best_peak_clearance, best_peak_time


class ScoringContext:
    """
    Holds everything that's shared across all objects being scored in
    one Tonight's Best request: equipment FOV, horizon profile,
    observing location, and the sample-time grid spanning tonight's
    darkness. Built once per request, then reused per object via
    score_object() to avoid re-deriving shared state.
    """

    def __init__(self, equipment, horizon, location, dusk, dawn):
        self.equipment = equipment
        self.horizon = horizon
        self.location = location
        self.dusk = dusk
        self.dawn = dawn

        self.sample_times = build_sample_times(dusk, dawn)
        self.sample_time_array = Time(list(self.sample_times))
        self.total_darkness_minutes = (dawn - dusk).total_seconds() / 60.0
        self._mid_window = self.sample_times[len(self.sample_times) // 2]

    def score_object(self, ra_deg, dec_deg, magnitude, size_str, object_type=None):
        """Compute the full ScoreBreakdown for one catalog object."""
        size_score = score_size_fit(
            size_str,
            self.equipment.fov_width_arcmin,
            self.equipment.fov_height_arcmin,
            object_type=object_type,
        )
        brightness_score = score_brightness(magnitude)

        run_minutes, peak_clearance_deg, peak_time = find_best_shooting_window(
            ra_deg, dec_deg, self.location, self.sample_times, self.sample_time_array, self.horizon
        )

        time_fraction = 0.0
        if self.total_darkness_minutes > 0:
            time_fraction = min(1.0, run_minutes / self.total_darkness_minutes)
        peak_clearance_fraction = max(0.0, min(1.0, peak_clearance_deg / 90.0))
        # Peak clearance only acts as a tiebreak between equal run lengths,
        # so it's scaled down enough to never outweigh a longer run.
        time_score = time_fraction + 0.001 * peak_clearance_fraction

        moon_reference_time = peak_time if peak_time is not None else self._mid_window
        moon_state = moon_state_at(Time(moon_reference_time), self.location)
        separation_deg = moon_state.separation_deg(ra_deg, dec_deg)
        moon_score = score_moon_distance(separation_deg, moon_state)

        composite = size_score * (
            BRIGHTNESS_WEIGHT * brightness_score
            + TIME_WEIGHT * time_score
            + MOON_WEIGHT * moon_score
        )

        return ScoreBreakdown(
            composite=composite,
            size_score=size_score,
            brightness_score=brightness_score,
            time_score=time_score,
            moon_score=moon_score,
            run_minutes=run_minutes,
            peak_clearance_deg=peak_clearance_deg,
        )
