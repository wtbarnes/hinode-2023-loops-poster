"""
Perform density diagnostic from EIS data
"""
import pathlib

import astropy.units as u
import numpy as np
from scipy.interpolate import interp1d
import xarray
import fiasco

__all__ = ['map_ratio_to_quantity']


def get_line_ratio(ion, density, numerator_transitions, denominator_transitions, **kwargs):
    """
    For a given ion and a range of densities, calculate the intensity ratio for a selected
    set of transitions at the formation temperature of the ion.
    """
    # Create a new Ion instance with a single temperature at the formation temperature
    ion = ion._new_instance(temperature=ion.formation_temperature)
    # Calculate contribution function
    g_of_nt = ion.contribution_function(density, **kwargs)
    # Get corresponding wavelengths
    w = ion.transitions.wavelength[~ion.transitions.is_twophoton]
    # Find indices corresponding to numerator and denominator
    # NOTE: these are the closest wavelengths to those specified.
    i_numerator = [np.argmin(np.fabs(w - t)) for t in numerator_transitions]
    i_denominator = [np.argmin(np.fabs(w - t)) for t in denominator_transitions]
    # Take ratio of sums over numerator and denominator transitions
    ratio = g_of_nt[..., i_numerator].sum(axis=2) / g_of_nt[..., i_denominator].sum(axis=2)
    return ratio.squeeze()


def map_ratio_to_quantity(observed_ratio, quantity, theoretical_ratio):
    """
    For a given intensity ratio curve and two associated intensity maps, calculate the resulting
    density map.
    """
    f_ratio_to_quantity = interp1d(theoretical_ratio.decompose().value,
                                   quantity.value,
                                   bounds_error=False,
                                   fill_value=np.nan)
    observed_quantity = f_ratio_to_quantity(observed_ratio.data)
    observed_quantity = xarray.DataArray(observed_quantity, coords=observed_ratio.coords)
    observed_quantity.attrs['unit'] = quantity.unit.to_string()
    return observed_quantity


if __name__ == '__main__':
    density = np.logspace(7, 12, 20) * u.Unit('cm-3')
    temperature = np.linspace(1, 2, 1000) * u.MK
    fe12 = fiasco.Ion('Fe XII', temperature)
    fe13 = fiasco.Ion('Fe XIII', temperature)

    # NOTE: This is a mapping for which transitions are needed for the numerator and
    # denominator for each density diagnostic
    transitions = {
        'Fe XII': {
            'numerator': [186.854, 186.887] * u.angstrom,
            'denominator': [195.119] * u.angstrom,
        },
        'Fe XIII': {
            'numerator': [203.795, 203.826] * u.angstrom,
            'denominator': [202.044] * u.angstrom,
        }
    }

    intensity_ratio_fe12 = get_line_ratio(fe12,
                                          density,
                                          transitions['Fe XII']['numerator'],
                                          transitions['Fe XII']['denominator'])
    intensity_ratio_fe13 = get_line_ratio(fe13,
                                          density,
                                          transitions['Fe XIII']['numerator'],
                                          transitions['Fe XIII']['denominator'])
    
    eis_ds = xarray.open_dataset(snakemake.input[0])
    observed_ratio_fe12 = eis_ds['Fe XII 186.880_intensity'] / eis_ds['Fe XII 195.119_intensity']
    observed_ratio_fe13 = eis_ds['Fe XIII 203.826_intensity'] / eis_ds['Fe XIII 202.044_intensity']

    observed_density_fe12 = map_ratio_to_quantity(observed_ratio_fe12, density, intensity_ratio_fe12)
    observed_density_fe12.attrs['transitions_numerator'] = transitions['Fe XII']['numerator'].to_value()
    observed_density_fe12.attrs['transitions_denominator'] = transitions['Fe XII']['denominator'].to_value()
    observed_density_fe13 = map_ratio_to_quantity(observed_ratio_fe13, density, intensity_ratio_fe13)
    observed_density_fe13.attrs['transitions_numerator'] = transitions['Fe XIII']['numerator'].to_value()
    observed_density_fe13.attrs['transitions_denominator'] = transitions['Fe XIII']['denominator'].to_value()

    observed_density = xarray.Dataset({
        'Fe XII': observed_density_fe12,
        'Fe XIII': observed_density_fe13,
    })
    parent_dir = pathlib.Path(snakemake.output[0]).parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    observed_density.to_netcdf(snakemake.output[0])
