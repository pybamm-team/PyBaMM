from pybamm import cosh, Parameter


def lico2_entropic_change_Moura2016(sto):
    """
    Lithium Cobalt Oxide (LiCO2) entropic change in open circuit potential (OCP) at
    a temperature of 298.15K as a function of the stochiometry. The fit is taken
    from Scott Moura's FastDFN code [1].

    References
    ----------
    .. [1] https://github.com/scott-moura/fastDFN

      Parameters
      ----------
      sto : :class:`pybamm.Symbol`
           Stochiometry of material (li-fraction)

    """
    c_p_max = Parameter("Maximum concentration in positive electrode [mol.m-3]")

    # Since the equation for LiCo2 from this ref. has the stretch factor,
    # should this too? If not, the "bumps" in the OCV don't line up.
    stretch = 1.062
    sto = stretch * sto

    du_dT = (
        0.07645 * (-54.4806 / c_p_max) * ((1.0 / cosh(30.834 - 54.4806 * sto)) ** 2)
        + 2.1581 * (-50.294 / c_p_max) * ((cosh(52.294 - 50.294 * sto)) ** (-2))
        + 0.14169 * (19.854 / c_p_max) * ((cosh(11.0923 - 19.8543 * sto)) ** (-2))
        - 0.2051 * (5.4888 / c_p_max) * ((cosh(1.4684 - 5.4888 * sto)) ** (-2))
        - (0.2531 / 0.1316 / c_p_max) * ((cosh((-sto + 0.56478) / 0.1316)) ** (-2))
        - (0.02167 / 0.006 / c_p_max) * ((cosh((sto - 0.525) / 0.006)) ** (-2))
    )

    return du_dT
