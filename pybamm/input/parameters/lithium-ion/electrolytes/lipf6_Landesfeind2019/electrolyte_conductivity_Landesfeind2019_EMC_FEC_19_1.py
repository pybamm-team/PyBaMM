import electrolyte_conductivity_Landesfeind2019_base as base
import numpy as np


def electrolyte_conductivity_Landesfeind2019_EMC_FEC_19_1(c_e, T):
    """
    Conductivity of LiPF6 in EMC:FEC (19:1) as a function of ion concentration and
    Temperature. The data comes from [1].
    References
    ----------
    .. [1] Landesfeind, J. and Gasteiger, H.A., 2019. Temperature and Concentration
    Dependence of the Ionic Transport Properties of Lithium-Ion Battery Electrolytes.
    Journal of The Electrochemical Society, 166(14), pp.A3079-A3097.
    ----------
    c_e: :class:`pybamm.Symbol`
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`
        Dimensional temperature
    Returns
    -------
    :class:`pybamm.Symbol`
        Electrolyte diffusivity
    """
    coeffs = np.array([2.51e-2, 1.75e2, 1.23, 2.05e-1, -8.81e-2, 2.83e-3])
    return base.electrolyte_conductivity_Landesfeind2019_base(c_e, T, coeffs)
