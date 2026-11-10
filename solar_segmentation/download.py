"""Downloading SDO/AIA and SDO/HMI FITS files from JSOC and picking one matched observation."""

import os
import time

import drms
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.io import fits
from astropy.time import Time
from sunpy.net import Fido, attrs as a

# Searches JSOC and downloads the files for one query, retrying on empty results or bad files
def search_and_fetch(query, name, data_dir, max_retries=3, min_file_size=1_000_000):
    for attempt in range(1, max_retries + 1):
        print(f"\n{name}: attempt {attempt}/{max_retries}...")
        try:
            result = Fido.search(query)
        except Exception as e:   # JSOC sometimes times out or drops the connection
            print(f"{name}: search failed ({type(e).__name__}: {e}), retrying...")
            time.sleep(5)
            continue
        n_results = len(result[0]) if len(result) else 0
        print(f"{name}: {n_results} results found")

        if n_results == 0:   # JSOC sometimes returns an empty search for a query that worked before
            print(f"{name}: no results, retrying search in 30 s...")
            time.sleep(30)
            continue

        try:
            files = Fido.fetch(result, path=f'{data_dir}/{{file}}', overwrite=False)   # reuse files already downloaded
        except Exception as e:   # JSOC export status checks can time out
            wait = 60 if 'pending export' in str(e) else 5   # JSOC allows one export per user at a time
            print(f"{name}: download failed ({type(e).__name__}: {e}), retrying in {wait} s...")
            time.sleep(wait)
            continue

        if getattr(files, 'errors', None):   # downloads that failed part-way
            print(f"{name}: {len(files.errors)} file(s) failed to download")

        good_files = []
        for f in files:
            if not os.path.isfile(f):          # failed downloads can come back as the folder path
                continue
            size = os.path.getsize(f)
            if size > min_file_size:
                good_files.append(f)
            else:
                print(f"  BAD (size={size} bytes): {f} -- removing")
                os.remove(f)

        if good_files:
            print(f"{name}: SUCCESS -- {len(good_files)} valid file(s)")
            return good_files

        print(f"{name}: no valid files, retrying...")
        time.sleep(5)

    print(f"{name}: FAILED after {max_retries} attempts")
    return []

# Converts a JSOC time string (e.g. 2014.10.24_00:36:00_TAI) to an astropy Time
def _parse_jsoc_time(s):
    date, clock = s.replace('_TAI', '').split('_')
    return Time(f"{date.replace('.', '-')}T{clock}", scale='tai')


# First HMI 720 s slot at or after time_start where both Ic and the magnetogram have data
def find_hmi_slot(time_start, search_hours=2):
    client = drms.Client()
    t_tai = Time(time_start).tai.strftime('%Y.%m.%d_%H:%M:%S') + '_TAI'
    good = None
    for series, seg in (('hmi.Ic_720s', 'continuum'), ('hmi.M_720s', 'magnetogram')):
        keys, segs = client.query(f'{series}[{t_tai}/{search_hours}h]', key='T_REC', seg=seg)
        has_data = segs[seg].astype(str).str.startswith('/').values   # missing slots read 'NoDataDirectory'
        slots = set(keys['T_REC'].values[has_data])
        good = slots if good is None else good & slots
    if not good:
        raise RuntimeError(f"No HMI Ic + magnetogram data within {search_hours} h of {time_start}")
    t_rec = min(good)   # JSOC time strings sort chronologically
    t_slot = _parse_jsoc_time(t_rec).utc
    print(f"HMI: first slot with Ic and magnetogram data is {t_rec} ({t_slot.isot} UTC)")
    return t_rec, t_slot


# Exports and downloads one exact HMI record through drms, reusing a file already in data_dir
def fetch_hmi(series, segment, t_rec, email, data_dir, max_retries=3, min_file_size=1_000_000):
    client = drms.Client(email=email)
    ds = f'{series}[{t_rec}]{{{segment}}}'
    for attempt in range(1, max_retries + 1):
        print(f"\n{ds}: attempt {attempt}/{max_retries}...")
        try:
            req = client.export(ds, method='url', protocol='fits')
            req.wait()
            if req.status != 0:
                raise RuntimeError(f"export status {req.status}")
        except Exception as e:   # JSOC export can time out
            print(f"{ds}: export failed ({type(e).__name__}: {e}), retrying...")
            time.sleep(5)
            continue

        if req.urls.empty or not all(req.urls['filename']):   # record exists but JSOC has no file to send
            print(f"{ds}: JSOC has no file for this record, retrying won't help -- try another TIME_START")
            return []

        good_files = []
        for i, row in req.urls.iterrows():
            path = os.path.join(data_dir, row['filename'])
            if not (os.path.isfile(path) and os.path.getsize(path) > min_file_size):
                try:
                    req.download(data_dir, index=i)
                except Exception as e:   # download can drop part-way
                    print(f"{ds}: download failed ({type(e).__name__}: {e})")
            if os.path.isfile(path) and os.path.getsize(path) > min_file_size:
                good_files.append(path)
        if good_files:
            print(f"{ds}: SUCCESS -- {len(good_files)} valid file(s)")
            return good_files
        print(f"{ds}: no valid files, retrying...")
        time.sleep(5)

    print(f"{ds}: FAILED after {max_retries} attempts")
    return []


# Downloads HMI Ic and magnetogram for the first good HMI slot, and AIA 1700 and 1600 within 30 s of it
def download_observation(time_start, email, data_dir, search_hours=2, max_retries=3, min_file_size=1_000_000):
    os.makedirs(data_dir, exist_ok=True)
    t_rec, t_slot = find_hmi_slot(time_start, search_hours)
    time_range = a.Time(t_slot - 30*u.s, t_slot + 30*u.s)

    files = {}
    for key, wl, name in (('aia1700', 1700, "AIA 1700"), ('aia1600', 1600, "AIA 1600")):
        query = (time_range & a.jsoc.Series('aia.lev1_uv_24s') & a.jsoc.Wavelength(wl*u.angstrom)
                 & a.jsoc.Notify(email))
        files[key] = sorted(search_and_fetch(query, name, data_dir, max_retries, min_file_size))
    files['ic']  = fetch_hmi('hmi.Ic_720s', 'continuum', t_rec, email, data_dir, max_retries, min_file_size)
    files['mag'] = fetch_hmi('hmi.M_720s', 'magnetogram', t_rec, email, data_dir, max_retries, min_file_size)

    failed = [key for key, f in files.items() if not f]
    if failed:
        raise RuntimeError(f"Download failed for {failed} -- check the log above")
    return files, t_slot


# Observation time from the header: T_REC (720 s window centre) for HMI, T_OBS (mid-exposure) for AIA
def _obs_time(path):
    with fits.open(path) as hdul:
        hdr = next((h.header for h in hdul if 'INSTRUME' in h.header), None)
    if hdr is None:
        raise KeyError(f"No INSTRUME keyword in {path}")
    if hdr['INSTRUME'].startswith('HMI'):
        return _parse_jsoc_time(hdr['T_REC'])
    return Time(hdr['T_OBS'].rstrip('Z'), scale='utc')


# Picks the file nearest t_ref for each channel, raising if any is more than max_gap seconds away;
# returns the chosen files and a table of their times
def select_observation(files, t_ref, max_gap=60):
    selected, rows = {}, []
    for key in ('ic', 'mag', 'aia1700', 'aia1600'):
        gaps = np.array([(_obs_time(p) - t_ref).sec for p in files[key]])
        i = int(np.argmin(np.abs(gaps)))
        selected[key] = files[key][i]
        rows.append({'Channel': key, 'Time (UTC)': (t_ref + gaps[i] * u.s).utc.isot,
                     'Gap (s)': round(gaps[i], 1), 'File': os.path.basename(selected[key])})
        if abs(gaps[i]) > max_gap:
            raise RuntimeError(f"{key} is {gaps[i]:.0f} s from the HMI slot (max_gap={max_gap} s)")
    return selected, pd.DataFrame(rows).set_index('Channel')