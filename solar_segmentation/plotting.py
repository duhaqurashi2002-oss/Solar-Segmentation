"""Interactive plotly tabs for checking each stage of the pipeline."""

import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import ipywidgets as widgets
from skimage import measure
import sunpy.visualization.colormaps  # registers the sdoaia colour maps with matplotlib


# Scales an image to 0-1 between vmin and vmax and colours it with a colormap
def _to_rgb(img, cmap, vmin, vmax):
    norm = np.clip((np.nan_to_num(img) - vmin) / (vmax - vmin), 0, 1)
    rgb = (plt.get_cmap(cmap)(norm)[..., :3] * 255).astype(np.uint8)
    rgb[~np.isfinite(img)] = 0
    return rgb


# Colours each pixel of the label map by its class colour in seg_classes
def _seg_to_rgb(seg_map, seg_classes):
    rgb = np.zeros(seg_map.shape + (3,), dtype=np.uint8)
    for val, col in seg_classes.values():
        rgb[seg_map == val] = [int(col[i:i+2], 16) for i in (1, 3, 5)]
    return rgb


# Makes a plotly figure from a downsampled, compressed image, keeping full-resolution pixel coordinates
def _image_fig(rgb, title, step, size, fmt='png'):
    H, W = rgb.shape[:2]
    fig = px.imshow(rgb[::step, ::step], binary_string=True, binary_format=fmt, origin='lower',
                    x=np.arange(0, W, step), y=np.arange(0, H, step))
    fig.update_layout(title=title, width=size, height=size,
                      margin=dict(l=10, r=10, t=70, b=10),
                      legend=dict(orientation='h', x=0, y=1.0, yanchor='bottom'))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    # Keep pixels square; a zoomed rectangle shrinks the plot area to its shape instead of stretching
    fig.update_xaxes(constrain='domain')
    fig.update_yaxes(scaleanchor='x', scaleratio=1, constrain='domain')
    return fig


# Draws the outline of a mask
def _add_contour(fig, mask, name, seg_classes, width, visible=True):
    col = seg_classes[name][1]
    contours = [measure.approximate_polygon(c, tolerance=0.5)
                for c in measure.find_contours(mask.astype(np.uint8), 0.5)]
    if not contours:
        return
    pts = np.concatenate([np.vstack([c, [[np.nan, np.nan]]]) for c in contours])
    fig.add_trace(go.Scatter(x=pts[:, 1], y=pts[:, 0], mode='lines', name=name,
                             line=dict(color=col, width=width), hoverinfo='skip',
                             visible=True if visible else 'legendonly'))


# Puts groups of plotly figures into tabs, figures in the same tab shown side by side
def image_tabs(groups):
    tab = widgets.Tab()
    children = []
    for figs in groups.values():
        items = []
        for fig in figs:
            if isinstance(fig, go.FigureWidget):
                items.append(fig)                 # interactive widget, shown directly
            else:
                out = widgets.Output()
                with out:
                    fig.show()
                items.append(out)
        children.append(widgets.HBox(items))
    tab.children = children
    for i, title in enumerate(groups):
        tab.set_title(i, title)
    return tab


# Links zoom and pan across figures: zooming one moves all the others to the same region
def link_zoom(figs):
    syncing = {'on': False}
    def sync(layout, xrange, yrange):
        if syncing['on']:
            return
        syncing['on'] = True
        try:
            for f in figs:
                if f.layout is layout:
                    continue
                with f.batch_update():
                    if xrange is not None:
                        f.layout.xaxis.range = xrange
                    if yrange is not None:
                        f.layout.yaxis.range = yrange
        finally:
            syncing['on'] = False
    for f in figs:
        f.layout.on_change(sync, 'xaxis.range', 'yaxis.range')


# Raw vs CLV-corrected, one tab per channel
def clv_tabs(channels, b_threshold, step, size):
    """channels: {tab title: (raw, corrected, cmap)}"""
    pair_size = size * 3 // 4
    groups = {}
    for name, (raw, corr, cmap) in channels.items():
        rmin, rmax = np.nanpercentile(raw, [1, 99])
        groups[name] = [
            _image_fig(_to_rgb(raw, cmap, rmin, rmax), f'Raw {name}', step, pair_size, fmt='jpg'),
            _image_fig(_to_rgb(corr, cmap, 0.5, 1.5), f'CLV-corrected {name} (B={b_threshold}G)',
                       step, pair_size, fmt='jpg')]
    return image_tabs(groups)


# One tab per image on the common grid (raw, before CLV correction)
def alignment_tabs(aia_rotated, aia1600_reprojected, ic_reprojected, mag_reprojected, step, size):
    def pct(img):
        return np.nanpercentile(img, [1, 99])
    return image_tabs({
        'AIA 1700': [_image_fig(_to_rgb(aia_rotated.data, 'sdoaia1700', *pct(aia_rotated.data)),
                                'Rotated AIA 1700', step, size, fmt='jpg')],
        'AIA 1600': [_image_fig(_to_rgb(aia1600_reprojected, 'sdoaia1600', *pct(aia1600_reprojected)),
                                'AIA 1600 reprojected onto AIA 1700 grid', step, size, fmt='jpg')],
        'HMI Ic':   [_image_fig(_to_rgb(ic_reprojected, 'gray', *pct(ic_reprojected)),
                                'HMI Ic reprojected onto AIA grid', step, size, fmt='jpg')],
        'Magnetogram': [_image_fig(_to_rgb(mag_reprojected, 'gray', -500, 500),
                                   'Magnetogram reprojected onto AIA grid', step, size, fmt='jpg')]})


# Builds four tabs: coloured segmentation, AIA 1700 with network shaded, AIA 1600 with plage outlined,
# HMI Ic with umbra, penumbra and pores outlined; zoom is linked across all four
def segmentation_tabs(seg_map, ic, aia_data, aia1600_data, masks, seg_classes, contour_width, step, size):
    fig_seg = _image_fig(_seg_to_rgb(seg_map, seg_classes), 'Combined segmentation', step, size)
    for name, (_, col) in seg_classes.items():
        fig_seg.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name=name,
                                     marker=dict(symbol='square', size=12, color=col)))

    aia_vmin, aia_vmax = np.nanpercentile(aia_data, [1, 99])
    aia_rgb = _to_rgb(aia_data, 'sdoaia1700', aia_vmin, aia_vmax)
    net = masks['network']
    net_col = np.array([int(seg_classes['network'][1][i:i+2], 16) for i in (1, 3, 5)])
    aia_rgb[net] = (0.4 * aia_rgb[net] + 0.6 * net_col).astype(np.uint8)
    fig_aia = _image_fig(aia_rgb, 'CLV-corrected AIA 1700: network', step, size, fmt='jpg')
    fig_aia.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name='network (shaded)',
                                 marker=dict(symbol='square', size=12, color=seg_classes['network'][1])))

    v1600_min, v1600_max = np.nanpercentile(aia1600_data, [1, 99])
    fig_1600 = _image_fig(_to_rgb(aia1600_data, 'sdoaia1600', v1600_min, v1600_max),
                          'CLV-corrected AIA 1600: plage', step, size, fmt='jpg')
    _add_contour(fig_1600, masks['plage'], 'plage', seg_classes, contour_width)

    fig_hmi = _image_fig(_to_rgb(ic, 'gray', 0.3, 1.3), 'CLV-corrected HMI Ic',
                         step, size, fmt='jpg')
    _add_contour(fig_hmi, masks['penumbra'], 'penumbra', seg_classes, contour_width)
    _add_contour(fig_hmi, masks['umbra'], 'umbra', seg_classes, contour_width)
    if 'pore' in masks:
        _add_contour(fig_hmi, masks['pore'], 'pore', seg_classes, contour_width)

    for fig in (fig_aia, fig_1600, fig_hmi):
        fig.update_layout(showlegend=True)

    # Interactive widgets so the zoom can be linked across tabs
    fig_seg, fig_aia, fig_1600, fig_hmi = [go.FigureWidget(f) for f in (fig_seg, fig_aia, fig_1600, fig_hmi)]
    link_zoom([fig_seg, fig_aia, fig_1600, fig_hmi])

    return image_tabs({'Segmentation': [fig_seg], 'AIA 1700': [fig_aia],
                       'AIA 1600': [fig_1600], 'HMI Ic': [fig_hmi]})
