from pybamm import tanh


def nmc_LGM50_ocp_Chen2020(sto):
    """
     LG M50 NMC open-circuit potential as a function of stoichiometry.  The fit is
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
        -0.809 * sto
        + 4.4875
        - 0.0428 * tanh(18.5138 * (sto - 0.5542))
        - 17.7326 * tanh(15.789 * (sto - 0.3117))
        + 17.5842 * tanh(15.9308 * (sto - 0.312))
    )

    return U
