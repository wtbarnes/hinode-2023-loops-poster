"""
Compute temperature diagnostic from AIA filter ratio
"""
import pathlib

import astropy.units as u
import numpy as np
import xarray
from synthesizAR.instruments.sdo import _TEMPERATURE_RESPONSE as AIA_RESPONSE

from density_diagnostic import map_ratio_to_quantity

if __name__ == '__main__':
    ds = xarray.open_dataset(snakemake.input[0])
    channel_a = '193'
    channel_b = '171'
    intensity_ratio = ds[channel_a] / ds[channel_b]
    T_lower = 0.5*u.MK
    T_upper = 2*u.MK
    T_indices = np.where(np.logical_and(AIA_RESPONSE['temperature'] >= T_lower,
                                        AIA_RESPONSE['temperature'] <= T_upper))
    filter_ratio = AIA_RESPONSE[channel_a] / AIA_RESPONSE[channel_b]
    filter_temperature = AIA_RESPONSE['temperature']
    temperature_map = map_ratio_to_quantity(intensity_ratio,
                                            filter_temperature[T_indices],
                                            filter_ratio[T_indices])
    temperature_map.attrs['numerator_filter'] = channel_a
    temperature_map.attrs['denominator_filter'] = channel_b
    parent_dir = pathlib.Path(snakemake.output[0]).parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    temperature_map.to_netcdf(snakemake.output[0])
