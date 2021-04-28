from electrolyte_base_Landesfeind2019 import electrolyte_TDF_base_Landesfeind2019
import numpy as np


def electrolyte_TDF_EC_DMC_1_1_Landesfeind2019(c_e, T):
    """
    Thermodynamic factor (TDF) of LiPF6 in EC:DMC (1:1 w:w) as a function of ion
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
        [-5.58, 7.17, 3.80e-2, 1.91, -6.65e-2, -5.08e-5, 1.1e-1, -6.10e-3, 1.51e-4]
    )

    return electrolyte_TDF_base_Landesfeind2019(c_e, T, coeffs)
