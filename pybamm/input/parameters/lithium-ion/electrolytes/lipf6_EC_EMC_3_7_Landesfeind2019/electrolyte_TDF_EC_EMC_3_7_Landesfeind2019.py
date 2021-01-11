from electrolyte_base_Landesfeind2019 import electrolyte_TDF_base_Landesfeind2019
import numpy as np


def electrolyte_TDF_EC_EMC_3_7_Landesfeind2019(c_e, T):
    """
    Thermodynamic factor (TDF) of LiPF6 in EC:EMC (3:7 w:w) as a function of ion
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
        Electrolyte thermodynamic factor
    """
    coeffs = np.array(
        [2.57e1, -4.51e1, -1.77e-1, 1.94, 2.95e-1, 3.08e-4, 2.59e-1, -9.46e-3, -4.54e-4]
    )

    return electrolyte_TDF_base_Landesfeind2019(c_e, T, coeffs)
