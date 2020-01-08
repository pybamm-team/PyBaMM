#
# Open-circuit voltage in the negative (lead) electrode
#
from pybamm import log10


def lead_ocp_Bode1977(m):
    """
    Dimensional open-circuit voltage in the negative (lead) electrode [V], from [1]_,
    as a function of the molar mass m [mol.kg-1].

    References
    ----------
    .. [1] H Bode. Lead-acid batteries. John Wiley and Sons, Inc., New York, NY, 1977.

    """
    U = (
        -0.294
        - 0.074 * log10(m)
        - 0.030 * log10(m) ** 2
        - 0.031 * log10(m) ** 3
        - 0.012 * log10(m) ** 4
    )
    return U
