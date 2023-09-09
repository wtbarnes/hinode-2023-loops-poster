"""
Normalize, correct, reproject, stack, and interpolate AIA data
"""
import pathlib

import numpy as np
from scipy.interpolate import interp1d
import dask
import dask.array
import distributed
import zarr

import astropy.io.fits
import astropy.wcs
import astropy.time
import astropy.units as u
from reproject import reproject_adaptive

import ndcube
import sunpy.map
from sunpy.map.header_helper import make_fitswcs_header
from sunpy.net import Fido, attrs as a
from sunpy.coordinates import propagate_with_solar_surface
from aiapy.calibrate import correct_degradation
from aiapy.calibrate.util import get_correction_table
import eispac.core

from net.heliocloud import HelioCloudClient
import net.attrs as heliocloud_attrs

AIA_SCALE = [0.6,  0.6]* u.arcsec / u.pix


def get_header(filename):
    with astropy.io.fits.open(filename, use_fsspec=True, fsspec_kwargs={'anon': True}, lazy_load_hdus=True) as hdul:
        header = hdul[1].header
    return astropy.io.fits.Header(header)


@dask.delayed
def get_data(filename):
    with astropy.io.fits.open(filename, use_fsspec=True, fsspec_kwargs={'anon': True}) as hdul:
        data = hdul[1].data
    return data


def map_from_delayed(delayed_array, header, dtype):
    shape = (header['NAXIS2'], header['NAXIS1'])
    array = dask.array.from_delayed(delayed_array, shape, dtype=dtype,)
    return sunpy.map.Map(array, header)


@dask.delayed
def reproject_array(input_array, input_wcs, target_wcs):
    with propagate_with_solar_surface():
        out_array = reproject_adaptive((input_array, input_wcs),
                                       target_wcs,
                                       shape_out=target_wcs.array_shape,
                                       roundtrip_coords=False,
                                       return_footprint=False)
    return out_array


def reproject_map(input_map, target_wcs):
    delayed_array = reproject_array(input_map.data, input_map.wcs, target_wcs)
    target_header = target_wcs.to_header()
    target_header['NAXIS1'] = target_wcs.array_shape[1]
    target_header['NAXIS2'] = target_wcs.array_shape[0]
    return map_from_delayed(delayed_array, target_header, input_map.data.dtype)


def stack_and_interpolate(cutouts, time_cutouts, time_common):
    # Stack data
    data_stacked = np.stack([c.data for c in cutouts], axis=0)
    # Rechunk along time axis
    data_stacked = data_stacked.rechunk(chunks=data_stacked.shape[:1]+(300, 300))
    # Interpolate to common time
    f_interp = lambda y: interp1d(time_cutouts.to_value('s'), y, axis=0, kind='linear', fill_value='extrapolate')(time_common.to_value('s'))
    data_interp = dask.array.map_blocks(
        f_interp,
        data_stacked,
        chunks=time_common.shape+data_stacked.chunks[1:],
        dtype=data_stacked.dtype
    )
    # Add the time axis to our coordinate system
    combined_wcs = cutouts[0].wcs.to_header()
    combined_wcs['CTYPE3'] = 'TIME'
    combined_wcs['CUNIT3'] = 's'
    combined_wcs['CDELT3'] = np.diff(time_common)[0].to_value('s')
    combined_wcs['CRPIX3'] = 1
    combined_wcs['CRVAL3'] = time_common[0].to_value('s')
    combined_wcs = astropy.wcs.WCS(combined_wcs)

    return ndcube.NDCube(data_interp, wcs=combined_wcs, unit=cutouts[0].unit, meta=cutouts[0].meta)


def build_cutout_cube(file_urls, time_common, ref_header, correction_table=None):
    ref_wcs = astropy.wcs.WCS(header=ref_header)
    # NOTE: this may be slow for lots of files
    all_headers = [get_header(url) for url in file_urls]
    
    full_disk_maps = [map_from_delayed(get_data(f), h, np.int32) for f,h in zip(file_urls, all_headers)]
    normalized_maps =[sunpy.map.Map(m.data/m.exposure_time.to_value('s'), m.meta) for m in full_disk_maps]
    corrected_maps = [correct_degradation(m, correction_table=correction_table) for m in normalized_maps]
    reprojected_maps = [reproject_map(m, ref_wcs) for m in corrected_maps]
    
    time_maps = astropy.time.Time([m.date for m in corrected_maps])
    
    cube = stack_and_interpolate(reprojected_maps,
                                 (time_maps - time_common[0]).to('s'),
                                 (time_common-time_common[0]).to('s'))
    return cube


if __name__ == '__main__':
    client = distributed.Client(address=snakemake.config['client_address'])
    
    # NOTE: this all should be done per wavelength
    eis_level_2_5_dir = pathlib.Path(snakemake.input[0])
    output_dir = pathlib.Path(snakemake.output[0]).parent
    output_dir.parent.mkdir(exist_ok=True, parents=True)
    channel = float(snakemake.params.channel) * u.AA
    time_start = astropy.time.Time(snakemake.config['aia_start_time'])
    time_end = astropy.time.Time(snakemake.config['aia_end_time'])
    
    # Get filenames for given wavelength
    query = Fido.search(
        a.Time(time_start, time_end),
        a.Wavelength(channel),
        heliocloud_attrs.Dataset('AIA'),
    )
    
    # Build target header
    m_eis_ref = sunpy.map.Map(list(eis_level_2_5_dir.glob('*.fe_12_195_119.2c-0.int.fits')))
    extent = u.Quantity(m_eis_ref.dimensions) * u.Quantity(m_eis_ref.scale)
    shape = tuple(np.ceil(extent / AIA_SCALE).to_value('pix').astype(int)[::-1])
    ref_header = make_fitswcs_header(shape,
                                     m_eis_ref.center,
                                     scale=AIA_SCALE,
                                     rotation_angle=0*u.deg,
                                     wavelength=channel,
                                     instrument='AIA',
                                     telescope='SDO/AIA',
                                     observatory='SDO',
                                     detector='AIA',
                                     unit=u.Unit('ct/s'))
    
    # Get common time
    cadence = 4 * u.minute
    interval = time_end - time_start
    time_common = time_start + np.arange(0, interval.to_value('minute'), cadence.to_value('minute')) * u.minute
    
    # Get correction table
    correction_table = get_correction_table()
    
    # Build cutout cube
    cutout_cube = build_cutout_cube(query[0]['URL'],
                                    time_common,
                                    ref_header,
                                    correction_table=correction_table)
    
    # Save cutout cube to file
    root = zarr.open(output_dir, mode='a')
    ds = root.create_dataset(snakemake.params.channel,
                             shape=cutout_cube.data.shape,
                             overwrite=True)
    ds.attrs['wcs'] = cutout_cube.wcs.to_header_string()
    ds.attrs['meta'] = ref_header
    ds[:] = cutout_cube.data.compute()
