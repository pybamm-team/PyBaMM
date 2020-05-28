from pybamm import exp, cosh


def graphite_entropic_change_Moura2016(sto, c_n_max):
    """
        Graphite entropic change in open circuit potential (OCP) at a temperature of
        298.15K as a function of the stochiometry taken from Scott Moura's FastDFN code
        [1].

        References
        ----------
        .. [1] https://github.com/scott-moura/fastDFN

          Parameters
          ----------
          sto : :class:`pybamm.Symbol`
               Stochiometry of material (li-fraction)

    """

    du_dT = (
        -1.5 * (120.0 / c_n_max) * exp(-120 * sto)
        + (0.0351 / (0.083 * c_n_max)) * ((cosh((sto - 0.286) / 0.083)) ** (-2))
        - (0.0045 / (0.119 * c_n_max)) * ((cosh((sto - 0.849) / 0.119)) ** (-2))
        - (0.035 / (0.05 * c_n_max)) * ((cosh((sto - 0.9233) / 0.05)) ** (-2))
        - (0.0147 / (0.034 * c_n_max)) * ((cosh((sto - 0.5) / 0.034)) ** (-2))
        - (0.102 / (0.142 * c_n_max)) * ((cosh((sto - 0.194) / 0.142)) ** (-2))
        - (0.022 / (0.0164 * c_n_max)) * ((cosh((sto - 0.9) / 0.0164)) ** (-2))
        - (0.011 / (0.0226 * c_n_max)) * ((cosh((sto - 0.124) / 0.0226)) ** (-2))
        + (0.0155 / (0.029 * c_n_max)) * ((cosh((sto - 0.105) / 0.029)) ** (-2))
    )

    return du_dT
