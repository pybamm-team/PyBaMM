#
# Standard parameters for SEI models
#

import pybamm

# --------------------------------------------------------------------------------------
# Dimensional parameters


def alpha(m_ratio, phi_s, phi_e, U_inner, U_outer):
    return pybamm.FunctionParameter(
        "Inner SEI reaction proportion", m_ratio, phi_s, phi_e, U_inner, U_outer
    )

