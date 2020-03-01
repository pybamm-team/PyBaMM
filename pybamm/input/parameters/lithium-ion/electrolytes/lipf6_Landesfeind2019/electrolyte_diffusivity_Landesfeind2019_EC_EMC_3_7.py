import electrolyte_diffusivity_Landesfeind2019_base as base
import numpy as np


def electrolyte_diffusivity_Landesfeind2019_EC_EMC_3_7(c_e, T, T_inf, E_k_e, R_g):
    """
    Diffusivity of LiPF6 in EC:EMC (3:7) as a function of ion concentration and
    Temperature. The data comes from [1].
    References
    ----------
    .. [1] Landesfeind, J. and Gasteiger, H.A., 2019. Temperature and Concentration
    Dependence of the Ionic Transport Properties of Lithium-Ion Battery Electrolytes.
    Journal of The Electrochemical Society, 166(14), pp.A3079-A3097.
    ----------
    c_e: :class: `numpy.Array`
        Dimensional electrolyte concentration
    T: :class: `numpy.Array`
        Dimensional temperature
    Returns
    -------
    :`numpy.Array`
        Electrolyte diffusivity
    """
    coeffs = np.array([1.01e3,
                       1.01,
                       -1.56e3,
                       -4.87e2])
    return base.electrolyte_diffusivity_Landesfeind2019_base(c_e, T, coeffs)
