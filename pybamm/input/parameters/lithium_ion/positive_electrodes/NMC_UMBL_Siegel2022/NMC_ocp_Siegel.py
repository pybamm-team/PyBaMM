import pybamm


def NMC_ocp_Siegel(sto):
    """
    Nickel Managanese Cobalt Oxide (NMC) Open Circuit Potential (OCP) as a
    function of the stochiometry. The fit is taken from Peyman MPM.

    References
    ----------
    Peyman MPM manuscript (to be submitted)

    Parameters
    ----------
    sto : :class:`pybamm.Symbol`
       Stochiometry of material (li-fraction)

    """

    u_eq = (
        4.3379
        - 1.7811 * sto
        + 1.6743 * (sto ** 2)
        - 1.4649 * (sto ** 3)
        + 2.0000 * (sto ** 4)
        - 1.7811 * (sto ** 5)
        - 4.6680 * pybamm.exp(72.5563 * sto - 74.192)
    )

    return u_eq


# if __name__ == "__main__":  # pragma: no cover
#     x = pybamm.linspace(0, 1)
#     pybamm.plot(x, NMC_ocp_PeymanMPM(x))
