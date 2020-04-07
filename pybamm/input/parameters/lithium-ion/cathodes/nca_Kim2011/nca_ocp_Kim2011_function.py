from pybamm import exp


def nca_ocp_Kim2011_function(sto):
    """
    NCA open-circuit potential (OCP) [1]. Fit in paper seems wrong to using
    nca_ocp_Kim2011_data.csv instead.
    References
    ----------
    .. [1] Kim, G. H., Smith, K., Lee, K. J., Santhanagopalan, S., & Pesaran, A.
    (2011). Multi-domain modeling of lithium-ion batteries encompassing
    multi-physics in varied length scales. Journal of The Electrochemical
    Society, 158(8), A955-A969.

    Parameters
    ----------
    sto : :class:`pybamm.Symbol`
       Stochiometry of material (li-fraction)

    """

    u_eq = (
        1.68 * sto ** 10
        - 2.222 * sto ** 9
        + 15.056 * sto ** 8
        - 23.488 * sto ** 7
        + 81.246 * sto ** 6
        - 344.566 * sto ** 5
        + 621.3475 * sto ** 4
        - 544.774 * sto ** 3
        + 264.427 * sto ** 2
        - 66.3691 * sto
        + 11.8058
        - 0.61386 * exp(5.8201 * sto ** 136.4)
    )

    return u_eq
