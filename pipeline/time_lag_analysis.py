"""
Run time lag analysis for all channel pairs
"""
import pathlib

import astropy.units as u
import xarray
from sunkit_image.time_lag import time_lag


if __name__ == '__main__':
    channel_pairs = [(94, 335),
                     (94, 171),
                     (94, 193),
                     (94, 131),
                     (94, 211),
                     (335, 131),
                     (335, 193),
                     (335, 211),
                     (335, 171),
                     (211, 131),
                     (211, 171),
                     (211, 193),
                     (193, 171),
                     (193, 131),
                     (171, 131),]
    ds = xarray.open_dataset(snakemake.input[0])

    time = u.Quantity(ds.time.data, ds['171'].attrs['time_unit'])
    lag_bounds = [-6, 6]*u.h
    spatial_coords = ds.drop_dims('time').coords
    tl_arrays = {}
    for ca, cb in channel_pairs:
        tl = time_lag(ds[f'{ca}'].data, ds[f'{cb}'].data, time, lag_bounds=lag_bounds)
        attrs = {'unit': tl.unit.to_string(), 'channel_a': ca, 'channel_b': cb}
        tl_arrays[f'{ca}_{cb}'] = xarray.DataArray(tl.value, coords=spatial_coords, attrs=attrs)

    tl_ds = xarray.Dataset(tl_arrays)
    parent_dir = pathlib.Path(snakemake.output[0]).parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    tl_ds.to_netcdf(snakemake.output[0])
