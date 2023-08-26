"""
Tools for selecting loop traces from images
"""
import numpy as np
from scipy.interpolate import splprep, splev

import astropy.units as u
from astropy.coordinates import SkyCoord

from sunpy.map.maputils import _bresenham as bresenham

__all__ = [
    'straight_loop_indices',
    'interpolate_hpc_coord',
]


def straight_loop_indices(coord, width, image_wcs, from_inner=False):
    """
    Return pixel indices corresponding to a straightened loop defined by
    `coord` and `width`

    Parameters
    -----------
    coord
    width
    image_wcs

    Returns
    --------
    loop_cut
    xs_cut
    indices
    """
    # Get width in pixel units
    width_px, width_py = image_wcs.world_to_pixel(
        SkyCoord(Tx=u.Quantity([0*width.unit, width]),
                 Ty=u.Quantity([0*width.unit, width]),
                 frame=coord.frame))
    width_px = np.diff(width_px)[0]
    width_py = np.diff(width_py)[0]
    # Find pixels between each loop segment
    px, py = image_wcs.world_to_pixel(coord)
    if from_inner:
        px, py = get_inner(px, py, width_px, width_py)
    px = np.round(px).astype(int)
    py = np.round(py).astype(int)
    loop_pix = []
    for i in range(px.shape[0]-1):
        b = bresenham(x1=px[i], y1=py[i], x2=px[i+1], y2=py[i+1])
        # Pop the last one, unless this is the final entry because the first point
        # of the next section will be the same
        if i < px.shape[0]-2:
            b = b[:-1]
        loop_pix.append(b)

    # For each loop segment, calculate direction
    direction = np.diff(np.array([px, py]).astype(float), axis=1)
    direction /= np.linalg.norm(direction, axis=0)
    # For each pixel in each segment, find the pixels corresponding to
    # the perpendicular cut with width_px
    indices = []
    for i, seg in enumerate(loop_pix):
        # Segments can be empty if the interpolated loop coordinates
        # are too dense. If ==0 cases are not skipped, the np.stack
        # call raises an exception.
        if len(seg) > 0:
            _indices = []
            for p in seg:
                px, py = _cross_section_endpoints(p, direction[:,i], width_px, width_py, from_inner)
                px = np.round(px).astype(int)
                py = np.round(py).astype(int)
                b = bresenham(x1=px[0], y1=py[0], x2=px[-1], y2=py[-1])
                _indices.append(b)
            indices.append(np.stack(_indices, axis=2))

    # Interpolate each perpendicular cut to make sure they have an equal number
    # of pixels
    n_xs = max([l.shape[0] for l in indices])
    if n_xs%2 == 0:
        # Always make this odd so that the "spine" of the loop corresponds to
        # a particular pixel and not an edge
        n_xs += 1
    s = np.linspace(0, 1, n_xs)
    indices_interp = []
    for seg in indices:
        for i in range(seg.shape[-1]):
            # Parametric interpolation in pixel space 
            xs = seg[:, :, i].T
            _s = np.append(0., np.linalg.norm(np.diff(xs, axis=1), axis=0).cumsum())
            _s /= np.sum(np.diff(_s))
            tck, _ = splprep(xs, u=_s)
            px, py = splev(s, tck)
            indices_interp.append(np.array([px, py]).T)
    indices_interp = np.round(np.array(indices_interp)).astype(int)

    i_mid = int((n_xs - 1)/2)
    loop_coord = image_wcs.pixel_to_world(indices_interp[:, i_mid, 0],
                                          indices_interp[:, i_mid, 1])
    data = u.Quantity([loop_coord.Tx, loop_coord.Ty]).to('arcsec').value
    loop_cut = np.append(0., np.linalg.norm(np.diff(data, axis=1), axis=0).cumsum()) * u.arcsec
    xs_cut = width * s

    return loop_cut, xs_cut, indices_interp


def _cross_section_endpoints(p0, direction, length_x, length_y, from_inner):
    """
    Given a point, direction, and length, find the endpoints
    of the line that is perpendicular to the line defined by
    `p0` and `direction`
    """
    angle = -np.arccos(direction[0]) * np.sign(direction[1])
    if from_inner:
        d = np.array([0, 1])
    else:
        d = np.array([-1, 1]) / 2 
    x = d * length_x * np.sin(angle)
    y = d * length_y * np.cos(angle)
    return p0[0] + x, p0[1] + y


def interpolate_hpc_coord(coord, n, **kwargs):
    """
    Parametric interpolation of a Helioprojective coordinate
    onto a uniform grid with ``n`` points.
    
    Parameters
    ----------
    coord
    n

    Returns
    -------
    coord_interp
    """
    splev_kwargs = kwargs.get('splev_kwargs', {})
    splprep_kwargs = kwargs.get('splprep_kwargs', {})
    data = u.Quantity([coord.Tx, coord.Ty]).to('arcsec').value
    s = np.append(0., np.linalg.norm(np.diff(data, axis=1), axis=0).cumsum())
    s /= np.sum(np.diff(s))
    s_new = np.linspace(0, 1, n)
    tck, _ = splprep(data, u=s, **splprep_kwargs)
    Tx, Ty = splev(s_new, tck, **splev_kwargs)
    return SkyCoord(Tx=Tx*u.arcsec, Ty=Ty*u.arcsec, frame=coord.frame)