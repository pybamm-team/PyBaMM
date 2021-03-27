from pybamm import exp


def graphite_ocp_Ramadass2004(sto):
    """
    Graphite Open Circuit Potential (OCP) as a function of the
    stochiometry (theta?). The fit is taken from Ramadass 2004.

    References
    ----------
    .. [1] P. Ramadass, Bala Haran, Parthasarathy M. Gomadam, Ralph White, and Branko
    N. Popov. "Development of First Principles Capacity Fade Model for Li-Ion Cells."
    (2004)
    """

    u_eq = (
        0.7222
        + 0.1387 * sto
        + 0.029 * (sto ** 0.5)
        - 0.0172 / sto
        + 0.0019 / (sto ** 1.5)
        + 0.2808 * exp(0.9 - 15 * sto)
        - 0.7984 * exp(0.4465 * sto - 0.4108)
    )

    return u_eq
