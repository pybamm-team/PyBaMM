from pybamm import exp, constants, Parameter


def graphite_diffusivity_Dualfoil1998(sto, T):
    """
    Graphite diffusivity as a function of stochiometry [1, 2, 3].

    References
    ----------
     .. [1] > Ai, W., Kraft, L., Sturm, J., Jossen, A., & Wu, B. (2020).
     Electrochemical Thermal-Mechanical Modelling of Stress Inhomogeneity in
     Lithium-Ion Pouch Cells. Journal of The Electrochemical Society, 167(1), 013512
      DOI: 10.1149/2.0122001JES.
     .. [2] > Rieger, B., Erhard, S. V., Rumpf, K., & Jossen, A. (2016).
     A new method to model the thickness change of a commercial pouch cell
     during discharge. Journal of The Electrochemical Society, 163(8), A1566-A1575.

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
        Electrode stochiometry
    T: :class:`pybamm.Symbol`
        Dimensional temperature, [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity [m2.s-1]
    """
    D_ref = 3.9 * 10 ** (-14)
    E_D_s = 5000
    T_ref = Parameter("Reference temperature [K]")
    arrhenius = exp(E_D_s / constants.R * (1 / T_ref - 1 / T))
    return D_ref * arrhenius
