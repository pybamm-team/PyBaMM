from pybamm import exp, tanh


def graphite_LGM50_ocp_Chen2020(sto):
    """
    LG M50 Graphite open-circuit potential as a function of stochiometry. The fit is
    taken from [1].

    References
    ----------
    .. [1] Chang-Hui Chen, Ferran Brosa Planella, Kieran O’Regan, Dominika Gastol, W.
    Dhammika Widanage, and Emma Kendrick. "Development of Experimental Techniques for
    Parameterization of Multi-scale Lithium-ion Battery Models." Journal of the
    Electrochemical Society 167 (2020): 080534.

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
       Electrode stochiometry

    Returns
    -------
    :class:`pybamm.Symbol`
       Open-circuit potential
    """

    U = (
        1.9793 * exp(-39.3631 * sto)
        + 0.2482
        - 0.0909 * tanh(29.8538 * (sto - 0.1234))
        - 0.04478 * tanh(14.9159 * (sto - 0.2769))
        - 0.0205 * tanh(30.4444 * (sto - 0.6103))
    )

    return U
