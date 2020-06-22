from electrolyte_base_Landesfeind2019 import (
    electrolyte_conductivity_base_Landesfeind2019,
)
import numpy as np


def electrolyte_conductivity_EC_DMC_1_1_Landesfeind2019(c_e, T):
    """
    Conductivity of LiPF6 in EC:DMC (1:1 w:w) as a function of ion concentration and
    temperature. The data comes from [1].

    References
    ----------
    .. [1] Landesfeind, J. and Gasteiger, H.A., 2019. Temperature and Concentration
    Dependence of the Ionic Transport Properties of Lithium-Ion Battery Electrolytes.
    Journal of The Electrochemical Society, 166(14), pp.A3079-A3097.

    Parameters
    ----------
    c_e: :class:`pybamm.Symbol`
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Electrolyte conductivity
    """
    coeffs = np.array([7.98e-1, 2.28e2, -1.22, 5.09e-1, -4e-3, 3.79e-3])

    return electrolyte_conductivity_base_Landesfeind2019(c_e, T, coeffs)
