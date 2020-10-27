def lico2_ocp_Ramadass2004(sto):
    """
    Lithium Cobalt Oxide (LiCO2) Open Circuit Potential (OCP) as a a function of the
    stochiometry. The fit is taken from Ramadass 2004. Stretch is considered the
    overhang area negative electrode / area positive electrode, in Ramadass 2002.

    References
    ----------
    .. [1] P. Ramadass, Bala Haran, Parthasarathy M. Gomadam, Ralph White, and Branko
    N. Popov. "Development of First Principles Capacity Fade Model for Li-Ion Cells."
    (2004)

    Parameters
    ----------
    sto : :class:`pybamm.Symbol`
       Stochiometry of material (li-fraction)

    """

    stretch = 1.13
    sto = stretch * sto

    u_eq = (
        -4.656
        + 88.669 * (sto ** 2)
        - 401.119 * (sto ** 4)
        + 342.909 * (sto ** 6)
        - 462.471 * (sto ** 8)
        + 433.434 * (sto ** 10)
    ) / (
        -1
        + 18.933 * (sto ** 2)
        - 79.532 * (sto ** 4)
        + 37.311 * (sto ** 6)
        - 73.083 * (sto ** 8)
        + 95.96 * (sto ** 10)
    )

    return u_eq
