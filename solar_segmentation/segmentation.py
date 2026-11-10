"""Segmentation into umbra, penumbra, pore, plage, network and quiet Sun."""

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# Removes connected regions smaller than min_size pixels
def _filter_by_size(mask, min_size):
    if not np.any(mask):
        return mask
    lbl, n = ndimage.label(mask)
    sizes = ndimage.sum(mask, lbl, range(1, n + 1))
    return np.isin(lbl, np.where(sizes >= min_size)[0] + 1)


# Umbra, penumbra and pores from CLV-corrected Ic, in three steps:
# 1. dark regions: smoothed Ic more than dark_sigma quiet-Sun sigmas below the quiet-Sun median
# 2. K-means (2 clusters) on region log-area, min Ic and mean |B|: larger-area cluster = sunspots, other = pores
# 3. K-means (2 clusters) on Ic of sunspot pixels: darker cluster = umbra, other = penumbra
def segment_spots(ic, mag, on_disk, smooth_sigma, dark_sigma, min_region, min_b, seed):
    # Step 1: smoothing washes out granulation lanes; off-disk filled with QS level so NaNs don't spread
    ic_smooth = gaussian_filter(np.where(on_disk, ic, 1.0), sigma=smooth_sigma)
    qs_median = np.median(ic_smooth[on_disk])
    qs_sigma = 1.4826 * np.median(np.abs(ic_smooth[on_disk] - qs_median))   # robust sigma from the MAD
    dark_cut = qs_median - dark_sigma * qs_sigma
    dark = on_disk & (ic_smooth < dark_cut)
    print(f"Quiet Sun Ic: median {qs_median:.3f}, sigma {qs_sigma:.4f} -> dark below {dark_cut:.3f}")

    lbl, n = ndimage.label(dark)
    idx = np.arange(1, n + 1)
    regions = pd.DataFrame({
        'Area (px)':    ndimage.sum(dark, lbl, idx),
        'Mean Ic':      ndimage.mean(ic, lbl, idx),
        'Min Ic':       ndimage.minimum(ic, lbl, idx),
        'Mean |B| (G)': ndimage.mean(np.abs(mag), lbl, idx)},
        index=pd.Index(idx, name='Region'))

    keep = (regions['Area (px)'] >= min_region) & (regions['Mean |B| (G)'] >= min_b)   # pores and spots are magnetic
    regions = regions[keep].sort_values('Area (px)', ascending=False)
    print(f"{len(regions)} dark regions of at least {min_region} px with mean |B| >= {min_b} G")

    # Step 2: sunspots vs pores
    if len(regions) >= 2:
        F = np.column_stack([np.log10(regions['Area (px)']), regions['Min Ic'], regions['Mean |B| (G)']])
        F = StandardScaler().fit_transform(F)
        lab = KMeans(n_clusters=2, init='k-means++', n_init=10, random_state=seed).fit_predict(F)
        spot_group = regions['Area (px)'].groupby(lab).mean().idxmax()
        regions['Class'] = np.where(lab == spot_group, 'sunspot', 'pore')
    else:
        regions['Class'] = 'sunspot'

    dark_px = on_disk & (ic < dark_cut)   # unsmoothed, drops the smoothing halo
    spots = np.isin(lbl, regions.index[regions['Class'] == 'sunspot']) & dark_px
    pore = np.isin(lbl, regions.index[regions['Class'] == 'pore']) & dark_px

    # Step 3: umbra vs penumbra
    umbra = np.zeros(ic.shape, dtype=bool)
    if spots.sum() >= 2:
        km = KMeans(n_clusters=2, init='k-means++', n_init=10, random_state=seed)
        lab = km.fit_predict(ic[spots].reshape(-1, 1).astype(np.float32))
        centres = km.cluster_centers_.ravel()
        umbra[spots] = lab == centres.argmin()
        print(f"Umbra/penumbra Ic centres: {np.sort(centres).round(3)}")
    penumbra = spots & ~umbra
    return umbra, penumbra, pore, regions


# Plage and network, in two steps (the bright counterpart of segment_spots):
# 1. bright regions: smoothed AIA 1700 more than bright_sigma quiet-Sun sigmas above the quiet-Sun median
# 2. K-means (2 clusters) on region log-area, mean AIA 1600 and mean |B|:
#    the cluster highest in all three = plage (regions >= min_plage_size), every other bright region = network
def segment_bright(aia1700, aia1600, mag, non_spot, smooth_sigma, bright_sigma,
                   min_network_size, min_plage_size, seed):
    # Step 1: spots and off-disk filled with QS level so they don't pull the smoothing down
    s1700 = gaussian_filter(np.where(non_spot, aia1700, 1.0), sigma=smooth_sigma)
    qs_median = np.median(s1700[non_spot])
    qs_sigma = 1.4826 * np.median(np.abs(s1700[non_spot] - qs_median))   # robust sigma from the MAD
    bright_cut = qs_median + bright_sigma * qs_sigma
    bright = _filter_by_size(non_spot & (s1700 >= bright_cut), min_network_size)
    print(f"Quiet Sun 1700: median {qs_median:.3f}, sigma {qs_sigma:.4f} -> bright above {bright_cut:.3f}")

    lbl, n = ndimage.label(bright)
    idx = np.arange(1, n + 1)
    regions = pd.DataFrame({
        'Area (px)':     ndimage.sum(bright, lbl, idx),
        'Mean AIA 1600': ndimage.mean(aia1600, lbl, idx),
        'Mean |B| (G)':  ndimage.mean(np.abs(mag), lbl, idx)},
        index=pd.Index(idx, name='Region'))
    regions['Class'] = 'network'

    # Step 2: plage vs network
    if len(regions) >= 2:
        F = StandardScaler().fit_transform(np.column_stack([np.log10(regions['Area (px)']),
                                                            regions['Mean AIA 1600'], regions['Mean |B| (G)']]))
        lab = KMeans(n_clusters=2, init='k-means++', n_init=10, random_state=seed).fit_predict(F)
        plage_group = pd.DataFrame(F).groupby(lab).mean().sum(axis=1).idxmax()   # highest in all three
        regions.loc[(lab == plage_group) & (regions['Area (px)'].values >= min_plage_size), 'Class'] = 'plage'

    plage = np.isin(lbl, regions.index[regions['Class'] == 'plage'])
    network = bright & ~plage
    print(f"{n} bright regions: {(regions['Class'] == 'plage').sum()} plage, "
          f"{(regions['Class'] == 'network').sum()} network")
    return plage, network, regions


# Writes every class into one label map; later classes overwrite earlier ones
def build_label_map(on_disk, network, plage, pore, penumbra, umbra, seg_classes):
    seg_map = np.full(on_disk.shape, seg_classes['off-disk'][0])
    for name, mask in (('QS', on_disk), ('network', network), ('plage', plage),
                       ('pore', pore), ('penumbra', penumbra), ('umbra', umbra)):
        seg_map[mask] = seg_classes[name][0]

    masks = {name: seg_map == val for name, (val, _) in seg_classes.items() if val >= 0}
    return seg_map, masks


# Pixel counts and percentage of disk per class
def pixel_counts(seg_map, seg_classes):
    counts = pd.DataFrame(
        [(name, val, (seg_map == val).sum()) for name, (val, _) in seg_classes.items() if val >= 0],
        columns=['Class', 'Label', 'Pixels']).set_index('Class')
    counts['% of disk'] = 100 * counts['Pixels'] / counts['Pixels'].sum()
    counts.loc['TOTAL'] = ['-', counts['Pixels'].sum(), 100.0]
    return counts