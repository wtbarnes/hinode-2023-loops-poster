import pathlib

import numpy as np
import xarray
import asdf

import astropy.units as u
import sunpy.map
import eispac.core

from loop_selection import straight_loop_indices, interpolate_hpc_coord


if __name__ == '__main__':
    eis_data_dir = pathlib.Path(snakemake.input[0])
    loop_width = float(snakemake.config['loop_width']) * u.arcsec
    
    with asdf.open(snakemake.config['traced_loop_file']) as af:
        traced_loop = af.tree['loop']
    traced_loop = interpolate_hpc_coord(traced_loop, 25)
    
    eis_intensity_files = sorted(eis_data_dir.glob('*int.fits'))
    eis_velocity_files = sorted(eis_data_dir.glob('*.vel.fits'))
    eis_maps = sunpy.map.Map(eis_intensity_files + eis_velocity_files)
    data_arrays = {}
    for m in eis_maps:
        s_parallel, s_perp, indices = straight_loop_indices(traced_loop, loop_width, m.wcs)
        m.meta.pop('keycomments')  # cannot serialize dicts to netcdf
        straight_loop = xarray.DataArray(
            m.data[indices[..., 1], indices[..., 0]],
            dims=['s_parallel', 's_perp'],
            coords={
                's_parallel': s_parallel.to_value('arcsec'),
                's_perp': s_perp.to_value('arcsec'),
            },
            attrs={**m.meta,
                   's_parallel_unit': 'arcsec',
                   's_perp_unit': 'arcsec'},
            name=f'straight_loop_eis_{m.meta["line_id"]}_{m.measurement}',
        )
        data_arrays[f'{m.meta["line_id"]}_{m.measurement}'] = straight_loop

    data_set = xarray.Dataset(data_arrays)
    data_set.to_netcdf(snakemake.output[0])
