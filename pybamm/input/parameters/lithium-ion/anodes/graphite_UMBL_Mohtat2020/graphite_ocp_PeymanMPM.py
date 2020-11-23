import pybamm


def graphite_ocp_PeymanMPM(sto):
    """
    Graphite Open Circuit Potential (OCP) as a function of the
    stochiometry. The fit is taken from Peyman MPM [1].

    References
    ----------
    .. [1] Peyman Mohtat et al, MPM (to be submitted)
    """

    u_eq = (
        0.063
        + 0.8 * pybamm.exp(-75 * (sto + 0.007))
        - 0.0120 * pybamm.tanh((sto - 0.127) / 0.016)
        - 0.0118 * pybamm.tanh((sto - 0.155) / 0.016)
        - 0.0035 * pybamm.tanh((sto - 0.220) / 0.020)
        - 0.0095 * pybamm.tanh((sto - 0.190) / 0.013)
        - 0.0145 * pybamm.tanh((sto - 0.490) / 0.020)
        - 0.0800 * pybamm.tanh((sto - 1.030) / 0.055)
    )

    return u_eq


# if __name__ == "__main__": # pragma: no cover
#     import matplotlib.pyplot as plt
#     import numpy as np

#     x = np.linspace(0, 1)
#     plt.plot(x, graphite_ocp_PeymanMPM(x))
#     plt.show()
