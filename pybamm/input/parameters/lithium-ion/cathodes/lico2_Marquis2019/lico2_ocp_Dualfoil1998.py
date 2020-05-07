from pybamm import tanh, Scalar


def lico2_ocp_Dualfoil1998(sto):
    """
    Lithium Cobalt Oxide (LiCO2) Open Circuit Potential (OCP) as a a function of the
    stochiometry. The fit is taken from Dualfoil [1]. Dualfoil states that the data
    was measured by Oscar Garcia 2001 using Quallion electrodes for 0.5 < sto < 0.99
    and by Marc Doyle for sto<0.4 (for unstated electrodes). We could not find any
    other records of the Garcia measurements. Doyles fits can be found in his
    thesis [2] but we could not find any other record of his measurments.

    References
    ----------
    .. [1] http://www.cchem.berkeley.edu/jsngrp/fortran.html
    .. [2] CM Doyle. Design and simulation of lithium rechargeable batteries,
           1995.

    Parameters
    ----------
    sto : :class:`pybamm.Symbol`
       Stochiometry of material (li-fraction)

    """

    stretch = 1.062
    sto = stretch * sto

    u_eq = (
        Scalar(2.16216, "[V]")
        + Scalar(0.07645, "[V]") * tanh(30.834 - 54.4806 * sto)
        + Scalar(2.1581, "[V]") * tanh(52.294 - 50.294 * sto)
        - Scalar(0.14169, "[V]") * tanh(11.0923 - 19.8543 * sto)
        + Scalar(0.2051, "[V]") * tanh(1.4684 - 5.4888 * sto)
        + Scalar(0.2531, "[V]") * tanh((-sto + 0.56478) / 0.1316)
        - Scalar(0.02167, "[V]") * tanh((sto - 0.525) / 0.006)
    )

    return u_eq
