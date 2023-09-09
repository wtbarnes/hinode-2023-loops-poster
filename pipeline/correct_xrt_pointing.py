"""
Correct XRT pointing using AIA 94 Å
"""
import copy
import pathlib

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.io.fits
import astropy.units as u
import astropy.wcs
import sunpy.map
from sunpy.net import Fido, attrs as a
from sunpy.coordinates import propagate_with_solar_surface
from aiapy.calibrate import register, update_pointing

from correct_eis_pointing import shift_pointing, cross_correlate_with_aia
from net.heliocloud import HelioCloudClient
import net.attrs as heliocloud_attrs


if __name__ == '__main__':
    xrt_filenames = sorted(pathlib.Path(snakemake.input[0]).glob('*.fits'))
    xrt_maps = sunpy.map.Map(xrt_filenames)
    # NOTE: Can grab any EIS map. Just using to get needed coordinates
    eis_map = sunpy.map.Map(sorted(pathlib.Path(snakemake.input[1]).glob('*.fits'))[0])
    
    # Query AIA 94 Å map closest to the XRT observations as this is what we are
    # cross-correlating with
    q = Fido.search(
        a.Wavelength(94*u.angstrom),
        a.Time(xrt_maps[0].date-2*u.min, xrt_maps[0].date+2*u.min),
        heliocloud_attrs.Dataset('AIA'),
    )
    with astropy.io.fits.open(q[0]['URL'][0], use_fsspec=True, fsspec_kwargs={'anon': True}) as hdul:
        aia_map = sunpy.map.Map(hdul[1].data, hdul[1].header)
    
    # Crop XRT data for CC as XRT FOV must be smaller than AIA
    # Also remove roll angle
    # The assumption here is that the pointing offset is signficantly larger
    # than the difference in observer locations.
    pad = 200 * u.arcsec
    blc = SkyCoord(Tx=eis_map.bottom_left_coord.Tx-pad, Ty=eis_map.bottom_left_coord.Ty-pad,
                   frame=eis_map.coordinate_frame)
    trc = SkyCoord(Tx=eis_map.top_right_coord.Tx+pad, Ty=eis_map.top_right_coord.Ty+pad,
                   frame=eis_map.coordinate_frame)
    with propagate_with_solar_surface():
        xrt_submaps = [m.rotate().submap(blc, top_right=trc) for m in xrt_maps]
        
    # Cross-correlate with AIA and correct pointing
    ref_coord = cross_correlate_with_aia(xrt_submaps[0],
                                         aia_map.resample(u.Quantity(xrt_maps[0].dimensions)))
    xrt_maps_corrected = [shift_pointing(m, ref_coord) for m in xrt_submaps]
    
    # Build new header and WCS that is cropped to the EIS FOV but maintains the XRT resolution
    extent = u.Quantity(eis_map.dimensions) * u.Quantity(eis_map.scale)
    shape = tuple(np.ceil(extent / u.Quantity(xrt_maps_corrected[0].scale)).to_value('pix').astype(int)[::-1])
    ref_header = sunpy.map.make_fitswcs_header(shape,
                                               eis_map.center,
                                               scale=u.Quantity(xrt_maps_corrected[0].scale),
                                               rotation_matrix=eis_map.rotation_matrix,
                                               wavelength=xrt_maps_corrected[0].wavelength,
                                               instrument=xrt_maps_corrected[0].instrument,
                                               telescope=xrt_maps_corrected[0].meta['telescop'],
                                               observatory=xrt_maps_corrected[0].observatory,
                                               detector=xrt_maps_corrected[0].detector,
                                               unit=xrt_maps_corrected[0].unit)
    ref_wcs = astropy.wcs.WCS(header=ref_header)
    
    # Reproject to WCS at time of EIS observation and save
    output_dir = pathlib.Path(snakemake.output[0])
    output_dir.mkdir(exist_ok=True, parents=True)
    for m, filename in zip(xrt_maps_corrected, xrt_filenames):
        with propagate_with_solar_surface():
            m_reprojected = m.reproject_to(ref_wcs, algorithm='adaptive')
        new_header = copy.deepcopy(ref_header)
        new_header['ec_fw1_'] = m.meta['ec_fw1_']
        new_header['ec_fw2_'] = m.meta['ec_fw2_']
        m_reprojected = sunpy.map.Map(m_reprojected.data, new_header)
        m_reprojected.save(output_dir / filename.name)
