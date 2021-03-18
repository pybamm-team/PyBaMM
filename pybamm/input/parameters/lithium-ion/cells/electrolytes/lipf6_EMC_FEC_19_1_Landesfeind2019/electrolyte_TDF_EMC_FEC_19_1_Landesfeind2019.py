from electrolyte_base_Landesfeind2019 import electrolyte_TDF_base_Landesfeind2019
import numpy as np


def electrolyte_TDF_EMC_FEC_19_1_Landesfeind2019(c_e, T):
    """
    Thermodyamic factor (TDF) of LiPF6 in EMC:FEC (19:1 w:w) as a function of ion
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
        [3.22, -1.01e1, -1.58e-2, 6.12, 2.96e-2, 2.42e-5, -2.22e-1, -1.57e-2, 6.30e-6]
    )

    return electrolyte_TDF_base_Landesfeind2019(c_e, T, coeffs)
