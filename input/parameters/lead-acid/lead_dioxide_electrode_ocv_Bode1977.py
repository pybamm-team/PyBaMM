#
# Open-circuit voltage in the positive (lead-dioxide) electrode
#
import numpy as np


def lead_dioxide_electrode_ocv_Bode1977(m):
    """
    Dimensional open-circuit voltage in the positive (lead-dioxide) electrode [V],
    from [1], as a function of the molar mass m [mol.kg-1].

    [1] H Bode. Lead-acid batteries. John Wiley and Sons, Inc., New York, NY, 1977.

    """
    U = (
        1.628
        + 0.074 * np.log10(m)
        + 0.033 * np.log10(m) ** 2
        + 0.043 * np.log10(m) ** 3
        + 0.022 * np.log10(m) ** 4
    )
    return U
