import pybamm


def graphite_ocp_Siegel(sto):
    """
    Graphite Open Circuit Potential (OCP) as a function of the
    stochiometry. The fit is taken from Peyman MPM [1].

    References
    ----------
    .. [1] Peyman Mohtat et al, MPM (to be submitted)
    """

    u_eq = (
        0.115
        + 0.8000 * pybamm.exp(-75 * (sto - 0.0013))
        - 0.0040 * pybamm.tanh((sto - 0.1503) / 0.0046)
        - 0.0450 * pybamm.tanh((sto - 0.1804) / 0.0500)
        - 0.0160 * pybamm.tanh((sto - 0.5291) / 0.0250)
        - 0.0500 * pybamm.tanh((sto - 1) / 0.0458)
    )

    return u_eq


# if __name__ == "__main__":  # pragma: no cover
#     x = pybamm.linspace(1e-10, 1 - 1e-10, 1000)
#     # pybamm.plot(x, graphite_ocp_PeymanMPM(x))
#     pybamm.plot(x, -1e-8 * pybamm.log(x / (1 - x)))
