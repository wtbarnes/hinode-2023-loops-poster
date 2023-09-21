"""
Basic utilities for setting up HYDRAD runs and using with synthesizAR
"""
import copy

import astropy.units as u
import numpy as np
from scipy.interpolate import interp1d

__all__ = ['config_from_bundle', 'configure_footpoint_heating']


def config_from_bundle(config, skeleton, include_expansion=False, include_gravity=True):
    new_config = copy.deepcopy(config)
    length = u.Quantity([l.length for l in skeleton.loops]).mean()
    # Calculate mean gravity and magnetic field profiles
    s_norm = np.linspace(0, 1, 50)
    b_mean = []
    g_mean = []
    for l in skeleton.loops:
        b_mean.append(interp1d(l.field_aligned_coordinate_norm, l.field_strength.to_value('G'),
                               fill_value='extrapolate')(s_norm))
        g_mean.append(interp1d(l.field_aligned_coordinate_norm, l.gravity.to_value('cm s-2'),
                               fill_value='extrapolate')(s_norm))

    b_mean = u.Quantity(b_mean, 'G').mean(axis=0)
    g_mean = u.Quantity(g_mean, 'cm s-2').mean(axis=0)
    # Set up config dictionary
    new_config['general']['loop_length'] = length
    new_config['initial_conditions']['heating_location'] = length / 2
    if include_gravity:
        new_config['general']['poly_fit_gravity'] = {
            'order': 6,
            'domains': [0, 1],
            'x': s_norm * length,
            'y': g_mean,
        }
    if include_expansion:
        new_config['general']['poly_fit_magnetic_field'] = {
            'order': 6,
            'domains': [0, 1],
            'x': s_norm * length,
            'y': b_mean,
        }
    return new_config


@u.quantity_input
def configure_footpoint_heating(config,
                                q_0: u.Unit('erg cm-3 s-1'),
                                sh_ratio=1e300,
                                deposition_ratio=0.5,
                                fp_ratio=0.5):
    """
    Deposit energy at both footpoints

    Parameters
    ----------
    q_0 (u.Unit): _description_
    sh_ratio (_type_): _description_
    deposition_ratio (_type_): _description_
    config (_type_): _description_
    fp_ratio (int, optional): _description_. Defaults to 0.
        0 corresponds to heating only at the left footpoint,
        1 only at the right

    Returns
    -------
    : `dict`
        Heating parameters for two events at each footpoint
    """
    t_rise = 30 * u.s
    L = config['general']['loop_length']
    deposition = deposition_ratio * L
    scale_height = sh_ratio * L
    q_L = q_0 * (1 - fp_ratio)
    q_R = q_0 * fp_ratio
    return [
        {'time_start': 0*u.s,
         'total_duration': config['general']['total_time']-t_rise,
         'rise_duration': t_rise,
         'decay_duration': 0*u.s,
         'location': deposition,
         'scale_height': scale_height,
         'rate': q_L},
        {'time_start': 0*u.s,
         'total_duration': config['general']['total_time']-t_rise,
         'rise_duration': t_rise,
         'decay_duration': 0*u.s,
         'location': L - deposition,
         'scale_height': scale_height,
         'rate': q_R},
    ]
