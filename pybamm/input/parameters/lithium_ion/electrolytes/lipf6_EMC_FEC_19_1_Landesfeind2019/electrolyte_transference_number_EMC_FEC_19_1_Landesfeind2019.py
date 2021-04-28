from electrolyte_base_Landesfeind2019 import (
    electrolyte_transference_number_base_Landesfeind2019,
)
import numpy as np


def electrolyte_transference_number_EMC_FEC_19_1_Landesfeind2019(c_e, T):
    """
    Transference number of LiPF6 in EMC:FEC (19:1 w:w) as a function of ion
    concentration and temperature. The data comes from [1].

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
        Electrolyte transference number
    """
    coeffs = np.array(
        [-1.22e1, -3.05, 8.38e-2, 1.78, 1.51e-3, -1.37e-4, -2.45e-2, -5.15e-3, 2.14e-5]
    )

    return electrolyte_transference_number_base_Landesfeind2019(c_e, T, coeffs)
