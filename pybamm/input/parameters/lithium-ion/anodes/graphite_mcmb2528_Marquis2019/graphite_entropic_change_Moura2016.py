from pybamm import exp, cosh, Scalar


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
        -1.5 * (Scalar(120.0, "[V.K-1.mol.m-3]") / c_n_max) * exp(-120 * sto)
        + (Scalar(0.0351, "[V.K-1.mol.m-3]") / (0.083 * c_n_max))
        * ((cosh((sto - 0.286) / 0.083)) ** (-2))
        - (Scalar(0.0045, "[V.K-1.mol.m-3]") / (0.119 * c_n_max))
        * ((cosh((sto - 0.849) / 0.119)) ** (-2))
        - (Scalar(0.035, "[V.K-1.mol.m-3]") / (0.05 * c_n_max))
        * ((cosh((sto - 0.9233) / 0.05)) ** (-2))
        - (Scalar(0.0147, "[V.K-1.mol.m-3]") / (0.034 * c_n_max))
        * ((cosh((sto - 0.5) / 0.034)) ** (-2))
        - (Scalar(0.102, "[V.K-1.mol.m-3]") / (0.142 * c_n_max))
        * ((cosh((sto - 0.194) / 0.142)) ** (-2))
        - (Scalar(0.022, "[V.K-1.mol.m-3]") / (0.0164 * c_n_max))
        * ((cosh((sto - 0.9) / 0.0164)) ** (-2))
        - (Scalar(0.011, "[V.K-1.mol.m-3]") / (0.0226 * c_n_max))
        * ((cosh((sto - 0.124) / 0.0226)) ** (-2))
        + (Scalar(0.0155, "[V.K-1.mol.m-3]") / (0.029 * c_n_max))
        * ((cosh((sto - 0.105) / 0.029)) ** (-2))
    )

    return du_dT
