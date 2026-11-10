"""Saving the segmentation as a FITS label map."""

import os

import numpy as np
from astropy.io import fits


SOURCE_KEYS = {'ic': 'SRC_IC', 'mag': 'SRC_MAG', 'aia1700': 'SRC_1700', 'aia1600': 'SRC_1600'}


# Writes the label map as int16 FITS on grid_map's WCS, with class names, source files
# and pipeline parameters in the header
def save_label_map(path, seg_map, grid_map, seg_classes, sources, params):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    header = grid_map.wcs.to_header()   # pixel grid, pointing, observer position, DATE-OBS
    header['RSUN_OBS'] = (grid_map.rsun_obs.value, '[arcsec] apparent solar radius')
    header['BUNIT'] = 'label'
    for name, (val, _) in seg_classes.items():
        header[f'LABEL{val}'] = (name, f'class for pixel value {val}')
    for key, f in sources.items():
        header[SOURCE_KEYS[key]] = os.path.basename(f)   # no comment: long file names fill the 80-char card
    for key, value in params.items():
        header.add_history(f'{key} = {value}')

    fits.PrimaryHDU(seg_map.astype(np.int16), header=header).writeto(path, overwrite=True)
    print(f"Saved {path}")