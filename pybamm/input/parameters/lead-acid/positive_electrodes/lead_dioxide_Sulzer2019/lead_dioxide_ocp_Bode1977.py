#
# Open-circuit voltage in the positive (lead-dioxide) electrode
#
from pybamm import log10, Scalar


def lead_dioxide_ocp_Bode1977(m):
    """
    Dimensional open-circuit voltage in the positive (lead-dioxide) electrode [V],
    from [1]_, as a function of the molar mass m [mol.kg-1].

    References
    ----------
    .. [1] H Bode. Lead-acid batteries. John Wiley and Sons, Inc., New York, NY, 1977.

    """
    m /= Scalar(1, "[mol.kg-1]")
    U = (
        1.628
        + 0.074 * log10(m)
        + 0.033 * log10(m) ** 2
        + 0.043 * log10(m) ** 3
        + 0.022 * log10(m) ** 4
    ) * Scalar(1, "[V]")
    return U
