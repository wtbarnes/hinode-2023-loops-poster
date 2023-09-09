"""
Background removal from straightened loops
"""
import numpy as np
import xarray


def subtract_background(cube, threshold=None):
    if threshold is not None:
        # Interpolate over masked values
        cube_thresh = xarray.where(cube <= threshold, np.nan, cube).interpolate_na('s_perp').interpolate_na('s_parallel')
    else:
        cube_thresh = cube
    c1 = (cube_thresh[..., -1] - cube_thresh[..., 0]) / (cube_thresh.s_perp[-1] - cube_thresh.s_perp[0]) 
    c0 = cube_thresh[..., 0] - c1 * cube_thresh.s_perp[0]
    bg = c1 * cube_thresh.s_perp + c0
    cube_no_bg = cube - bg
    return xarray.where(cube_no_bg<0, np.nan, cube_no_bg)


if __name__ == '__main__':
    ds = xarray.open_dataset(snakemake.input[0])
    data_arrays = {}
    for k in ds.keys():
        data_arrays[k] = subtract_background(ds[k], threshold=0.0)
    
    ds_no_bg = xarray.Dataset(data_arrays)
    ds_no_bg.to_netcdf(snakemake.output[0])