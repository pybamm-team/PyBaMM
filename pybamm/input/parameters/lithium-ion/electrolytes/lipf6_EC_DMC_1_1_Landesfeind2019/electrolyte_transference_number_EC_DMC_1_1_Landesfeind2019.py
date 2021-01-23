from electrolyte_base_Landesfeind2019 import (
    electrolyte_transference_number_base_Landesfeind2019,
)
import numpy as np


def electrolyte_transference_number_EC_DMC_1_1_Landesfeind2019(c_e, T):
    """
    Transference number of LiPF6 in EC:DMC (1:1 w:w) as a function of ion
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
        [
            -7.91,
            2.45e-1,
            5.28e-2,
            6.98e-1,
            -1.08e-2,
            -8.21e-5,
            7.43e-4,
            -2.22e-3,
            3.07e-5,
        ]
    )

    return electrolyte_transference_number_base_Landesfeind2019(c_e, T, coeffs)
