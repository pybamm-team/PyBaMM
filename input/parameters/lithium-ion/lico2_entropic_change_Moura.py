import autograd.numpy as np


def lico2_entropic_change_Moura(sto, c_p_max):
    """
        Lithium Cobalt Oxide (LiCO2) entropic change in open circuit potential (OCP) at
        a temperature of 298.15K as a function of the stochiometry. The fit is taken
        from Scott Moura's FastDFN code [1].

        References
        ----------
        .. [1] https://github.com/scott-moura/fastDFN

          Parameters
          ----------
          sto: double
               Stochiometry of material (li-fraction)

    """

    du_dT = (
        0.07645 * (-54.4806 / c_p_max) * ((1.0 / np.cosh(30.834 - 54.4806 * sto)) ** 2)
        + 2.1581 * (-50.294 / c_p_max) * ((np.cosh(52.294 - 50.294 * sto)) ** (-2))
        + 0.14169 * (19.854 / c_p_max) * ((np.cosh(11.0923 - 19.8543 * sto)) ** (-2))
        - 0.2051 * (5.4888 / c_p_max) * ((np.cosh(1.4684 - 5.4888 * sto)) ** (-2))
        - (0.2531 / 0.1316 / c_p_max) * ((np.cosh((-sto + 0.56478) / 0.1316)) ** (-2))
        - (0.02167 / 0.006 / c_p_max) * ((np.cosh((sto - 0.525) / 0.006)) ** (-2))
    )

    return du_dT
