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
        4.396
        - 1.538 * sto
        + 0.7194 * (sto ** 2)
        - 0.009979 * (sto ** 3)
        + 1.074 * (sto ** 4)
        - 1.075 * (sto ** 5)
        - 4.071 * pybamm.exp(75 * sto - 80.9)
    )

    return u_eq


# if __name__ == "__main__":  # pragma: no cover
#     x = pybamm.linspace(0, 1)
#     pybamm.plot(x, NMC_ocp_PeymanMPM(x))
