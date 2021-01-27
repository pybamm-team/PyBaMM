import pybamm


def NMC_entropic_change_PeymanMPM(sto):
    """
    Nickel Manganese Cobalt (NMC) entropic change in open circuit potential (OCP) at
    a temperature of 298.15K as a function of the OCP. The fit is taken from [1].

    References
    ----------
    .. [1] W. Le, I. Belharouak, D. Vissers, K. Amine, "In situ thermal study of
    li1+ x [ni1/ 3co1/ 3mn1/ 3] 1- x o2 using isothermal micro-clorimetric
    techniques",
    J. of the Electrochemical Society 153 (11) (2006) A2147–A2151.

    Parameters
    ----------
    sto : :class:`pybamm.Symbol`
        Stochiometry of material (li-fraction)

    """

    # Since the equation uses the OCP at each stoichiometry as input,
    # we need OCP function here

    u_eq = (
        4.3452
        - 1.6518 * sto
        + 1.6225 * sto ** 2
        - 2.0843 * sto ** 3
        + 3.5146 * sto ** 4
        - 0.5623 * 10 ** (-4) * pybamm.exp(109.451 * sto - 100.006)
    )

    du_dT = (
        -800 + 779 * u_eq - 284 * u_eq ** 2 + 46 * u_eq ** 3 - 2.8 * u_eq ** 4
    ) * 10 ** (-3)

    return du_dT
