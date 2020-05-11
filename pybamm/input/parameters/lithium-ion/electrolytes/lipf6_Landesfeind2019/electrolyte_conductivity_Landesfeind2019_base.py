from pybamm import exp, sqrt


def electrolyte_conductivity_Landesfeind2019_base(c_e, T, coeffs):
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
    p1, p2, p3, p4, p5, p6 = coeffs
    A = p1 * (1 + (T - p2))
    B = 1 + p3 * sqrt(c) + p4 * (1 + p5 * exp(1000 / T)) * c
    C = 1 + c ** 4 * (p6 * exp(1000 / T))
    sigma_e = A * c * B / C  # mS.cm-1

    return sigma_e / 10
