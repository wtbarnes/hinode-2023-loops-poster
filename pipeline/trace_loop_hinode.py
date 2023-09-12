import pathlib

import xarray
import asdf
import zarr

import astropy.units as u
import astropy.wcs
import sunpy.map
import eispac.core

from loop_selection import straight_loop_indices, interpolate_hpc_coord

PRESERVED_KEYS = [
    'instrume',
    'telescop',
    'obsrvtry',
    'detector',
    'line_id',
    'measrmnt',
    'bunit',
    'ec_fw1_',
    'ec_fw2_',
]


if __name__ == '__main__':
    data_dir = pathlib.Path(snakemake.input[0])
    loop_width = float(snakemake.config['loop_width']) * u.arcsec
    
    with asdf.open(snakemake.config['traced_loop_file']) as af:
        traced_loop = af.tree['loop']
    traced_loop = interpolate_hpc_coord(traced_loop, 25)
    
    # Create AIA WCS that we want to reproject to
    aia_data_dir = pathlib.Path(snakemake.input[1])
    root = zarr.open(aia_data_dir.parent, mode='r')
    aia_wcs = astropy.wcs.WCS(root[aia_data_dir.name].attrs['meta'])  # All channels have same WCS
    
    if 'EIS' in snakemake.input[0]:
        glob_pattern = '*.int.fits'
    else:
        glob_pattern = '*.fits'
    maps = sunpy.map.Map(sorted(data_dir.glob(glob_pattern)))
    data_arrays = {}
    for m in maps:
        m_rep = m.reproject_to(aia_wcs, algorithm='adaptive')
        s_parallel, s_perp, indices = straight_loop_indices(traced_loop, loop_width, m_rep.wcs)
        coords={
            's_parallel': s_parallel.to_value('arcsec'),
            's_perp': s_perp.to_value('arcsec'),
        }
        key = m.measurement
        if line_id := m.meta.get('line_id', None):
            key = f'{line_id}_{key}'
        straight_loop = xarray.DataArray(
            m_rep.data[indices[..., 1], indices[..., 0]],
            dims=['s_parallel', 's_perp'],
            coords=coords,
            attrs={**{k: m.meta[k] for k in PRESERVED_KEYS if k in m.meta},
                   's_parallel_unit': 'arcsec',
                   's_perp_unit': 'arcsec'},
            name=f'straight_loop_{m.instrument}_{key}',
        )
        data_arrays[key] = straight_loop

    data_set = xarray.Dataset(data_arrays, coords=coords)
    data_set.to_netcdf(snakemake.output[0])
