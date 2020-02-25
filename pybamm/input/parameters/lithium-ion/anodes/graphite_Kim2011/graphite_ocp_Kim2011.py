from pybamm import exp, tanh


def graphite_ocp_Kim2011(sto):
    """
       Graphite Open Circuit Potential (OCP) as a function of the stochiometry [1].

       References
       ----------
       .. [1] Kim, G. H., Smith, K., Lee, K. J., Santhanagopalan, S., & Pesaran, A.
       (2011). Multi-domain modeling of lithium-ion batteries encompassing
       multi-physics in varied length scales. Journal of The Electrochemical
       Society, 158(8), A955-A969.
       """

    u_eq = (
        0.124
        + 1.5 * exp(-70 * sto)
        - 0.0351 * tanh((sto - 0.286) / 0.083)
        - 0.0045 * tanh((sto - 0.9) / 0.119)
        - 0.035 * tanh((sto - 0.99) / 0.05)
        - 0.0147 * tanh((sto - 0.5) / 0.034)
        - 0.102 * tanh((sto - 0.194) / 0.142)
        - 0.022 * tanh((sto - 0.98) / 0.0164)
        - 0.011 * tanh((sto - 0.124) / 0.0226)
        + 0.0155 * tanh((sto - 0.105) / 0.029)
    )

    return u_eq
