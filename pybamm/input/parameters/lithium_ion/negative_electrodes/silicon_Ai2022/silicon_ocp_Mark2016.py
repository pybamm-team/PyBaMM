from pybamm import LithiumIonParameters, Parameter, sigmoid


def silicon_ocp_Mark2016(sto):
    """
    silicon Open Circuit Potential (OCP) as a a function of the
    stochiometry. The fit is taken from the Enertech cell [1], which is only accurate
    for 0 < sto < 1.

    References
    ----------
    .. [1] Verbrugge M, Baker D, Xiao X. Formulation for the treatment of multiple
    electrochemical reactions and associated speciation for the Lithium-Silicon
    electrode[J]. Journal of The Electrochemical Society, 2015, 163(2): A262.

    Parameters
    ----------
    sto: double
       Stochiometry of material (li-fraction)

    Returns
    -------
    :class:`pybamm.Symbol`
        OCP [V]
    """
    current = LithiumIonParameters().dimensional_current_with_time
    capacity = Parameter("Nominal cell capacity [A.h]")
    k = 100
    m1 = sigmoid(current / capacity, 0, k)  # for lithation (current < 0)
    # m2 = sigmoid(-current / capacity, 0, k)  # for delithiation (current > 0)
    m2 = 1 - m1

    p1 = -96.63 * m1 - 51.02 * m2
    p2 = 372.6 * m1 + 161.3 * m2
    p3 = -587.6 * m1 - 205.7 * m2
    p4 = 489.9 * m1 + 140.2 * m2
    p5 = -232.8 * m1 - 58.76 * m2
    p6 = 62.99 * m1 + 16.87 * m2
    p7 = -9.286 * m1 - 3.792 * m2
    p8 = 0.8633 * m1 + 0.9937 * m2

    u_eq = (
        p1 * sto ** 7
        + p2 * sto ** 6
        + p3 * sto ** 5
        + p4 * sto ** 4
        + p5 * sto ** 3
        + p6 * sto ** 2
        + p7 * sto
        + p8
    )
    return u_eq


# below is another implemmentation
# p1 = -96.63
# p2 = 372.6
# p3 = -587.6
# p4 = 489.9
# p5 = -232.8
# p6 = 62.99
# p7 = -9.286
# p8 = 0.8633

# u_eq1 = (
#     p1 * sto ** 7
#     + p2 * sto ** 6
#     + p3 * sto ** 5
#     + p4 * sto ** 4
#     + p5 * sto ** 3
#     + p6 * sto ** 2
#     + p7 * sto
#     + p8
# )

# p1 =  - 51.02
# p2 = 161.3
# p3 = - 205.7
# p4 = 140.2
# p5 =  - 58.76
# p6 =  16.87
# p7 = - 3.792
# p8 = 0.9937

# u_eq2 = (
#     p1 * sto ** 7
#     + p2 * sto ** 6
#     + p3 * sto ** 5
#     + p4 * sto ** 4
#     + p5 * sto ** 3
#     + p6 * sto ** 2
#     + p7 * sto
#     + p8
# )
# return u_eq1 * m1 + u_eq2 * m2
