from pybamm import exp


def electrolyte_diffusivity_Landesfeind2019_base(c_e, T, coeffs):
    """
    Conductivity of LiPF6 in solvent_X as a function of ion concentration and
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
    coeffs: :class: `numpy.Array`
        Fitting parameter coefficients
    Returns
    -------
    :`numpy.Array`
        Electrolyte diffusivity
    """
    c = c_e / 1000  # mol.m-3 -> mol.l
    p1, p2, p3, p4 = coeffs
    A = p1 * exp(p2 * c)
    B = exp(p3 / T)
    C = exp(p4 * c / T)
    D_e = A * B * C * 1e-6  # m2/s

    return D_e
