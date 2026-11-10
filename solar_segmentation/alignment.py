"""Putting every channel on one common pixel grid."""

from reproject import reproject_interp


# Rotates raw AIA 1700 to solar north (order=3), then reprojects raw AIA 1600, HMI Ic and the
# magnetogram onto that grid
def align_to_aia(aia_map, aia1600_map, ic_map, mag_map):
    aia_rotated = aia_map.rotate(order=3)
    shape = aia_rotated.data.shape
    print("Rotated AIA shape:", shape)

    aia1600_reprojected, _ = reproject_interp(aia1600_map, aia_rotated.wcs, shape_out=shape)
    ic_reprojected, _ = reproject_interp(ic_map, aia_rotated.wcs, shape_out=shape)
    mag_reprojected, _ = reproject_interp(mag_map, aia_rotated.wcs, shape_out=shape)
    return aia_rotated, aia1600_reprojected, ic_reprojected, mag_reprojected