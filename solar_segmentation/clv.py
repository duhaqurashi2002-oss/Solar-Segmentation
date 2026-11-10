"""Centre-to-limb variation (CLV) correction."""

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord


# Divides the image by a polynomial in mu fitted to quiet-Sun pixels (|B| below b_threshold);
# image and mag must both be on grid_map's pixel grid
def clv_correct(grid_map, image, mag, b_threshold, poly_order):
    ny, nx = image.shape
    y, x = np.mgrid[0:ny, 0:nx]
    centre = SkyCoord(0*u.arcsec, 0*u.arcsec, frame=grid_map.coordinate_frame)
    x0, y0 = grid_map.wcs.world_to_pixel(centre)
    r_sun_px = (grid_map.rsun_obs / grid_map.scale.axis1).to_value(u.pix)
    r = np.sqrt((x - x0)**2 + (y - y0)**2)
    mu = np.sqrt(np.clip(1 - (r / r_sun_px)**2, 0, 1))
    on_disk = r <= r_sun_px

    quiet_sun_mask = (np.abs(mag) < b_threshold) & on_disk & np.isfinite(image)
    coeffs = np.polyfit(mu[quiet_sun_mask], image[quiet_sun_mask], poly_order)
    ctl_profile = np.poly1d(coeffs)(mu)
    ctl_profile[~on_disk] = np.nan
    corrected = image / ctl_profile
    return corrected, ctl_profile, quiet_sun_mask
