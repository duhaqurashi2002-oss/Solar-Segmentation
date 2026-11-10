# Solar-Segmentation

Pixel-wise segmentation of SDO full-disk images into umbra, penumbra, pore, plage, network and quiet Sun, combining HMI continuum intensity, the HMI line-of-sight magnetogram, and AIA 1600 Å and 1700 Å.

Each feature type is found the same way: first detect what stands out from the quiet Sun, then let K-means decide what it is.

## Installation

```
git clone https://github.com/duhaqurashi2002-oss/Solar-Segmentation.git
cd Solar-Segmentation
pip install -r requirements.txt
```

Downloading from JSOC needs an email address registered at http://jsoc.stanford.edu/ajax/register_email.html.

## Usage

Open `main.ipynb`, set your email and an observation time in section 2.1, and run all cells:

```python
JSOC_EMAIL = "youremail@hotmail.com"
TIME_START = '2014-10-22T00:00:00'
```

The method parameters in section 2.2 default to the values used in the paper.

## Pipeline

1. **Download:** finds the first HMI 720 s slot with continuum and magnetogram data at or after `TIME_START`, then fetches AIA 1700 Å and 1600 Å within 30 s of it.
2. **Alignment:** rotates AIA 1700 Å to solar north and reprojects AIA 1600 Å, HMI Ic and the magnetogram onto that grid.
3. **CLV correction:** divides each image by a polynomial in μ fitted to quiet-Sun pixels (|B| < 50 G).
4. **Segmentation:**
   - **Umbra, penumbra, pore:** dark regions in smoothed HMI Ic (5σ below the quiet-Sun median); K-means on region area, minimum Ic and mean |B| splits sunspots from pores; K-means on Ic splits sunspot pixels into umbra and penumbra.
   - **Plage, network:** bright regions in smoothed AIA 1700 Å (3σ above the quiet-Sun median); K-means on region area, mean AIA 1600 Å and mean |B| splits plage from network.
   - **Quiet Sun:** everything else on the disk.
5. **Output:** a FITS label map in `output/`.

## Output

`output/segmentation_YYYYMMDD_HHMMSS.fits` is a 16-bit integer image on the rotated AIA grid, with the WCS in the header so it opens directly as a `sunpy.map.Map`.

| Value | Class |
|---|---|
| -1 | off-disk |
| 0 | quiet Sun |
| 1 | network |
| 2 | plage |
| 3 | penumbra |
| 4 | umbra |
| 5 | pore |

The header also records the class names (`LABEL0` to `LABEL5`), the four source files (`SRC_IC`, `SRC_MAG`, `SRC_1700`, `SRC_1600`) and every pipeline parameter (`HISTORY`).

## Repository layout

```
main.ipynb               runs the full pipeline on one observation
solar_segmentation/
    download.py          JSOC search, download and file matching
    alignment.py         rotation and reprojection onto one grid
    clv.py               centre-to-limb variation correction
    segmentation.py      spot and bright-feature segmentation
    plotting.py          interactive plots for checking each stage
    output.py            FITS label map
Archive/                 dissertation notebooks (earlier pipeline versions)
```
