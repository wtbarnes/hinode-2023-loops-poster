import pathlib

import zarr
import asdf
import xarray

import astropy.wcs
import astropy.units as u
import ndcube

from loop_selection import straight_loop_indices, interpolate_hpc_coord

if __name__ == '__main__':
    aia_level_2_dir = pathlib.Path(snakemake.input[0]).parent
    loop_width = float(snakemake.config['loop_width']) * u.arcsec
    
    with asdf.open(snakemake.config['traced_loop_file']) as af:
        traced_loop = af.tree['loop']
    traced_loop = interpolate_hpc_coord(traced_loop, 25)
    
    root = zarr.open(aia_level_2_dir, mode='r')
    data_arrays = {}
    for channel in snakemake.params.channels:
        ds = root[channel]
        cube = ndcube.NDCube(
            ds[:],
            wcs=astropy.wcs.WCS(ds.attrs['wcs']),
            unit=ds.attrs['unit'],
            meta=ds.attrs['meta'],
        )
        s_parallel, s_perp, indices = straight_loop_indices(traced_loop, loop_width, cube[0].wcs)
        time = (cube.axis_world_coords(0)[0] - cube.axis_world_coords(0)[0][0])
        straight_loop = xarray.DataArray(
            cube.data[:, indices[..., 1], indices[..., 0]],
            dims=['time', 's_parallel', 's_perp'],
            coords={
                'time': time.to_value('s'),
                's_parallel': s_parallel.to_value('arcsec'),
                's_perp': s_perp.to_value('arcsec'),
            },
            attrs={**cube.meta,
                   's_parallel_unit': 'arcsec',
                   's_perp_unit': 'arcsec',
                   'time_unit': 's'},
            name=f'straight_loop_aia_{channel}',
        )
        data_arrays[channel] = straight_loop

    data_set = xarray.Dataset(data_arrays)
    data_set.to_netcdf(snakemake.output[0])
