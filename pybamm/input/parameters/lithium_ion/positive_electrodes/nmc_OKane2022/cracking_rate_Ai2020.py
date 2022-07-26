from pybamm import constants, exp


def cracking_rate_Ai2020(T_dim):
    """
    Particle cracking rate as a function of temperature [1, 2].

    References
    ----------
     .. [1] > Ai, W., Kraft, L., Sturm, J., Jossen, A., & Wu, B. (2020).
     Electrochemical Thermal-Mechanical Modelling of Stress Inhomogeneity in
     Lithium-Ion Pouch Cells. Journal of The Electrochemical Society, 167(1), 013512
      DOI: 10.1149/2.0122001JES.
     .. [2] > Deshpande, R., Verbrugge, M., Cheng, Y. T., Wang, J., & Liu, P. (2012).
     Battery cycle life prediction with coupled chemical degradation and fatigue
     mechanics. Journal of the Electrochemical Society, 159(10), A1730.

    Parameters
    ----------
    T: :class:`pybamm.Symbol`
        temperature, [K]

    Returns
    -------
    k_cr: :class:`pybamm.Symbol`
        cracking rate, [m/(Pa.m0.5)^m_cr]
        where m_cr is another Paris' law constant
    """
    k_cr = 3.9e-20
    Eac_cr = 0  # to be implemented
    arrhenius = exp(Eac_cr / constants.R * (1 / T_dim - 1 / 298.15))
    return k_cr * arrhenius
