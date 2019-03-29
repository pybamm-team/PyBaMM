#
# Single Particle Model (SPM)
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class SPM(pybamm.BaseModel):
    """Single Particle Model (SPM) of a lithium-ion battery.

    Attributes
    ----------

    rhs: dict
        A dictionary that maps expressions (variables) to expressions that represent
        the rhs
    initial_conditions: dict
        A dictionary that maps expressions (variables) to expressions that represent
        the initial conditions
    boundary_conditions: dict
        A dictionary that maps expressions (variables) to expressions that represent
        the boundary conditions
    variables: dict
        A dictionary that maps strings to expressions that represent
        the useful variables

    """

    def __init__(self):
        super().__init__()

        "Model Variables"
        # Particle concentration
        c_s_n = pybamm.Variable(
            "Negative particle concentration", domain="negative particle"
        )
        c_s_p = pybamm.Variable(
            "Positive particle concentration", domain="positive particle"
        )

        "Model Parameters and functions"
        sp = pybamm.standard_parameters
        spli = pybamm.standard_parameters_lithium_ion
        # Current function
        # i_cell = sp.current_with_time

        "Interface Conditions"
        G_n = pybamm.interface.homogeneous_reaction(["negative electrode"])
        G_p = pybamm.interface.homogeneous_reaction(["positive electrode"])

        "Model Equations"
        self.update(
            pybamm.particle.Standard(c_s_n, G_n), pybamm.particle.Standard(c_s_p, G_p)
        )

        "Additional Conditions"
        additional_bcs = {}
        self._boundary_conditions.update(additional_bcs)

        "Additional Model Variables"
        c_s_n_surf = pybamm.surf(c_s_n)
        c_s_p_surf = pybamm.surf(c_s_p)
        j0_n = sp.m_n * c_s_n_surf ** 0.5 * (1 - c_s_n_surf) ** 0.5
        j0_p = sp.m_p * spli.gamma_hat_p * c_s_p_surf ** 0.5 * (1 - c_s_p_surf) ** 0.5

        # linearise BV for now
        ocp = spli.U_p(c_s_p_surf) - spli.U_n(c_s_n_surf)
        Lambda = 38  # TODO: change this
        reaction_overpotential = -(2 / Lambda) * (1 / (j0_p * sp.lp)) - (2 / Lambda) * (
            1 / (j0_n * sp.ln)
        )
        voltage = ocp + reaction_overpotential

        additional_variables = {
            "negative particle surface concentration": c_s_n_surf,
            "positive particle surface concentration": c_s_p_surf,
            "voltage": voltage,
        }
        self._variables.update(additional_variables)
