"""
Local horizon profile: parses a Stellarium-style polygonal horizon
(.hrz) file and answers "what's the minimum clear altitude at this
azimuth?" via linear interpolation between the defined points.

No file is bundled by default. HORIZON_FILENAME in core/config.py
names a single server-wide horizon file (matching the single-user
deployment model already used for the catalog CSV); if that file is
absent, HorizonProfile.load() falls back to a flat 0 degree horizon
so the rest of the app keeps working.
"""
from core.config import logger, DEFAULT_HORIZON_ALTITUDE_DEG


class HorizonProfile:
    """
    A polygonal horizon defined by (azimuth_deg, altitude_deg) points,
    sorted by azimuth and wrapping around at the 360/0 boundary.
    """

    def __init__(self, points):
        """
        points: iterable of (azimuth_deg, altitude_deg) tuples.
        Must contain at least one point.
        """
        if not points:
            raise ValueError("HorizonProfile requires at least one point")
        # Normalise azimuth into [0, 360) and sort/dedupe by azimuth.
        normalised = sorted((az % 360.0, alt) for az, alt in points)
        deduped = []
        for az, alt in normalised:
            if deduped and deduped[-1][0] == az:
                deduped[-1] = (az, alt)
            else:
                deduped.append((az, alt))
        self._points = deduped

    @classmethod
    def flat(cls, min_altitude_deg=DEFAULT_HORIZON_ALTITUDE_DEG):
        """A trivial horizon that is level all the way around."""
        return cls([(0.0, min_altitude_deg), (180.0, min_altitude_deg)])

    @classmethod
    def from_hrz_text(cls, text):
        """
        Parse Stellarium .hrz polygonal horizon text.

        Format: an optional first line giving the point count, then one
        "azimuth altitude" pair per line (whitespace separated, degrees).
        Blank lines and lines starting with '#' or ';' are ignored. The
        point-count line is detected as a lone integer and skipped.
        """
        points = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith(";"):
                continue
            parts = line.split()
            if len(parts) == 1:
                # Point-count header line - not needed for parsing, skip.
                continue
            if len(parts) < 2:
                continue
            try:
                az = float(parts[0])
                alt = float(parts[1])
            except ValueError:
                logger.warning(f"Skipping unparsable horizon line: {raw_line!r}")
                continue
            points.append((az, alt))

        if not points:
            raise ValueError("No horizon points found in .hrz content")
        return cls(points)

    @classmethod
    def load(cls, path):
        """
        Load a horizon profile from a .hrz file path. Falls back to a
        flat DEFAULT_HORIZON_ALTITUDE_DEG horizon (with a warning) if
        the file is missing or unparsable, so callers never need to
        special-case "no file".
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            profile = cls.from_hrz_text(text)
            logger.info(f"Loaded horizon profile from {path} ({len(profile._points)} points)")
            return profile
        except FileNotFoundError:
            logger.info(
                f"No horizon file at {path}; using flat {DEFAULT_HORIZON_ALTITUDE_DEG} deg horizon"
            )
            return cls.flat(DEFAULT_HORIZON_ALTITUDE_DEG)
        except Exception as e:
            logger.warning(
                f"Failed to parse horizon file {path}: {e}; "
                f"using flat {DEFAULT_HORIZON_ALTITUDE_DEG} deg horizon"
            )
            return cls.flat(DEFAULT_HORIZON_ALTITUDE_DEG)

    def min_altitude_at(self, azimuth_deg):
        """
        Return the minimum clear altitude (degrees) at the given
        azimuth, linearly interpolating between the two nearest defined
        points and wrapping around the 360/0 boundary.
        """
        az = azimuth_deg % 360.0
        points = self._points

        if len(points) == 1:
            return points[0][1]

        # Find the bracketing points, wrapping past the end of the list.
        for i in range(len(points)):
            az_a, alt_a = points[i]
            az_b, alt_b = points[(i + 1) % len(points)]

            span = (az_b - az_a) % 360.0
            if span == 0:
                continue
            offset = (az - az_a) % 360.0
            if offset <= span:
                fraction = offset / span
                return alt_a + (alt_b - alt_a) * fraction

        # Shouldn't be reachable given the modular scan above.
        return points[0][1]
