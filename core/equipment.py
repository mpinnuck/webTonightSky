"""
Imaging equipment settings for a session (OTA focal length, aperture,
sensor size) and the field-of-view they derive. No Flask or scoring
concerns live here - this module only knows how to turn "what's
attached to the telescope" into "how big a patch of sky it sees".

Named/reusable equipment profiles (e.g. "Q150P + ASI2600") are managed
client-side (browser localStorage, with export/import for moving them
between devices) rather than stored on the server, since the server
has no concept of separate users - see client/static/index.html. The
browser sends the selected profile's raw specs inline as "equipment"
on the /api/list_objects Tonight's Best request, and this class is
what turns that into a field of view.
"""

ARCSEC_PER_RADIAN = 206264.80625
ARCMIN_PER_RADIAN = ARCSEC_PER_RADIAN / 60.0


class EquipmentSettings:
    """
    Session equipment settings and derived field of view.

    Sensor dimensions can be supplied either directly in millimetres
    (sensor_width_mm / sensor_height_mm) or as a pixel count plus pixel
    size (sensor_width_px / sensor_height_px / pixel_size_um), matching
    however the person prefers to enter their camera's spec sheet.
    Exactly one of the two forms must be supplied per axis.
    """

    def __init__(
        self,
        focal_length_mm,
        aperture_mm,
        sensor_width_mm=None,
        sensor_height_mm=None,
        sensor_width_px=None,
        sensor_height_px=None,
        pixel_size_um=None,
    ):
        if focal_length_mm is None or focal_length_mm <= 0:
            raise ValueError("focal_length_mm must be a positive number")
        if aperture_mm is None or aperture_mm <= 0:
            raise ValueError("aperture_mm must be a positive number")

        self.focal_length_mm = float(focal_length_mm)
        self.aperture_mm = float(aperture_mm)

        self.sensor_width_mm = self._resolve_dimension_mm(
            "sensor width", sensor_width_mm, sensor_width_px, pixel_size_um
        )
        self.sensor_height_mm = self._resolve_dimension_mm(
            "sensor height", sensor_height_mm, sensor_height_px, pixel_size_um
        )

    @staticmethod
    def _resolve_dimension_mm(label, dimension_mm, dimension_px, pixel_size_um):
        if dimension_mm is not None:
            dimension_mm = float(dimension_mm)
            if dimension_mm <= 0:
                raise ValueError(f"{label} must be a positive number")
            return dimension_mm

        if dimension_px is not None and pixel_size_um is not None:
            dimension_px = float(dimension_px)
            pixel_size_um = float(pixel_size_um)
            if dimension_px <= 0 or pixel_size_um <= 0:
                raise ValueError(f"{label} pixel count and pixel size must be positive")
            return dimension_px * pixel_size_um / 1000.0

        raise ValueError(
            f"{label} must be supplied either in mm or as pixel count + pixel_size_um"
        )

    @property
    def focal_ratio(self):
        """The f-number (focal length / aperture)."""
        return self.focal_length_mm / self.aperture_mm

    @property
    def fov_width_arcmin(self):
        """Field of view width, in arcminutes."""
        return (self.sensor_width_mm / self.focal_length_mm) * ARCMIN_PER_RADIAN

    @property
    def fov_height_arcmin(self):
        """Field of view height, in arcminutes."""
        return (self.sensor_height_mm / self.focal_length_mm) * ARCMIN_PER_RADIAN

    @classmethod
    def from_request_dict(cls, data):
        """
        Build an EquipmentSettings from a request JSON dict, e.g.:
            {
              "focal_length_mm": 800,
              "aperture_mm": 200,
              "sensor_width_mm": 23.5,
              "sensor_height_mm": 15.6
            }
        or using the pixel-based form:
            {
              "focal_length_mm": 800,
              "aperture_mm": 200,
              "sensor_width_px": 6248,
              "sensor_height_px": 4176,
              "pixel_size_um": 3.76
            }
        """
        return cls(
            focal_length_mm=data.get("focal_length_mm"),
            aperture_mm=data.get("aperture_mm"),
            sensor_width_mm=data.get("sensor_width_mm"),
            sensor_height_mm=data.get("sensor_height_mm"),
            sensor_width_px=data.get("sensor_width_px"),
            sensor_height_px=data.get("sensor_height_px"),
            pixel_size_um=data.get("pixel_size_um"),
        )
