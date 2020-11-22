from pybamm import exp


def LFP_ocp_ashfar2017(sto):
    """
    Open-circuit potential for LFP

    References
    ----------
    .. [1] Afshar, S., Morris, K., & Khajepour, A. (2017). Efficient electrochemical
    model for lithium-ion cells. arXiv preprint arXiv:1709.03970.

    Parameters
    ----------
    sto : :class:`pybamm.Symbol`
       Stochiometry of material (li-fraction)

    """

    c1 = -150 * sto
    c2 = -30 * (1 - sto)
    k = 3.4077 - 0.020269 * sto + 0.5 * exp(c1) - 0.9 * exp(c2)

    return k
