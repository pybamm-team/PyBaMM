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
    c_e: :class:`pybamm.Symbol`
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`
        Dimensional temperature
    coeffs: :class:`pybamm.Symbol`
        Fitting parameter coefficients
    Returns
    -------
    :class:`pybamm.Symbol`
        Electrolyte diffusivity
    """
    c = c_e / 1000  # mol.m-3 -> mol.l
    p1, p2, p3, p4 = coeffs
    A = p1 * exp(p2 * c)
    B = exp(p3 / T)
    C = exp(p4 * c / T)
    D_e = A * B * C * 1e-10  # m2/s

    return D_e
