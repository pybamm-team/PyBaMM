from pybamm import exp


def LFP_ocp_ashfar2017(sto):
    """
    exchange-current density for Butler-Volmer reactions
    References
    ----------

Efficient electrochemical model for lithium-ion cells
Sepideh Afshar, Kirsten Morris, Amir Khajepour

    ----------
    T: :class:`pybamm.Symbol`
        Dimensional temperature
    Returns
    -------
    :class:`pybamm.Symbol`
        exchange-current density
    """

    c1 = -150 * sto
    c2 = -30 * (1 - sto)
    k = 3.4077 - 0.020269 * sto + 0.5 * exp(c1) - 0.9 * exp(c2)

    return k
