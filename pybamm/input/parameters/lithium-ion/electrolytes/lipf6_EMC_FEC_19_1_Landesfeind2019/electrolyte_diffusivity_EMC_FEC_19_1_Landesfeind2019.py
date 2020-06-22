from electrolyte_base_Landesfeind2019 import (
    electrolyte_diffusivity_base_Landesfeind2019,
)
import numpy as np


def electrolyte_diffusivity_EMC_FEC_19_1_Landesfeind2019(c_e, T):
    """
    Diffusivity of LiPF6 in EMC:FEC (19:1 w:w) as a function of ion concentration and
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
        Electrolyte diffusivity
    """
    coeffs = np.array([5.86e2, 1.33, -1.38e3, -5.82e2])

    return electrolyte_diffusivity_base_Landesfeind2019(c_e, T, coeffs)
