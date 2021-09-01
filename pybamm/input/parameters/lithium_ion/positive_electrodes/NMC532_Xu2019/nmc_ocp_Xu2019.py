import pybamm


def nmc_ocp_Xu2019(sto):
    """
    Nickel Managanese Cobalt Oxide (NMC) Open Circuit Potential (OCP) as a
    function of the stochiometry, from [1].

    References
    ----------
    .. [1] Xu, Shanshan, Chen, Kuan-Hung, Dasgupta, Neil P., Siegel, Jason B. and
    Stefanopoulou, Anna G. "Evolution of Dead Lithium Growth in Lithium Metal Batteries:
    Experimentally Validated Model of the Apparent Capacity Loss." Journal of The
    Electrochemical Society 166.14 (2019): A3456-A3463.

    Parameters
    ----------
    sto : :class:`pybamm.Symbol`
       Stochiometry of material (li-fraction)

    """

    # Values from Mohtat2020, might be more accurate
    u_eq = (
        4.3452
        - 1.6518 * sto
        + 1.6225 * (sto ** 2)
        - 2.0843 * (sto ** 3)
        + 3.5146 * (sto ** 4)
        - 2.2166 * (sto ** 5)
        - 0.5623e-4 * pybamm.exp(109.451 * sto - 100.006)
    )

    # # only valid in range ~(0.25,0.95)
    # u_eq = (
    #     5744.862289 * sto ** 9
    #     - 35520.41099 * sto ** 8
    #     + 95714.29862 * sto ** 7
    #     - 147364.5514 * sto ** 6
    #     + 142718.3782 * sto ** 5
    #     - 90095.81521 * sto ** 4
    #     + 37061.41195 * sto ** 3
    #     - 9578.599274 * sto ** 2
    #     + 1409.309503 * sto
    #     - 85.31153081
    #     - 0.0003 * pybamm.exp(7.657 * sto ** 115)
    # )

    return u_eq
