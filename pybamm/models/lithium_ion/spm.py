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

        "Parameters"
        param = pybamm.standard_parameters
        param.__dict__.update(pybamm.standard_parameters_lithium_ion.__dict__)

        "Model Variables"
        # Particle concentration
        c_s_n = pybamm.Variable(
            "Negative particle concentration", domain="negative particle"
        )
        c_s_p = pybamm.Variable(
            "Positive particle concentration", domain="positive particle"
        )

        "Submodels"
        # Interfacial current density
        j_n = pybamm.interface.homogeneous_reaction(["negative electrode"])
        j_p = pybamm.interface.homogeneous_reaction(["positive electrode"])

        # Particle models
        negative_particle_model = pybamm.particle.Standard(c_s_n, j_n, param)
        positive_particle_model = pybamm.particle.Standard(c_s_p, j_p, param)

        "Combine Submodels"
        self.update(negative_particle_model, positive_particle_model)

        "Additional Conditions"
        additional_bcs = {}
        self._boundary_conditions.update(additional_bcs)

        "Additional Useful Variables"
        # surface concentrations
        c_s_n_surf = pybamm.surf(c_s_n)
        c_s_p_surf = pybamm.surf(c_s_p)

        # r-averaged concentrations

        # electrolyte potential

        # open circuit voltage

        # reaction overpotentials

        # j0_n = sp.m_n * c_s_n_surf ** 0.5 * (1 - c_s_n_surf) ** 0.5
        # j0_p = sp.m_p * sp.gamma_hat_p * c_s_p_surf ** 0.5 * (1 - c_s_p_surf) ** 0.5

        # linearise BV for now
        ocp = param.U_p(c_s_p_surf) - param.U_n(c_s_n_surf)
        # Lambda = 38  # TODO: change this
        # reaction_overpotential
        # = -(2 / Lambda) * (1 / (j0_p * sp.lp)) - (2 / Lambda) * (
        #     1 / (j0_n * sp.ln)
        # )
        terminal_voltage = ocp

        additional_variables = {
            "negative particle surface concentration": c_s_n_surf,
            "positive particle surface concentration": c_s_p_surf,
            "open circuit voltage": ocp,
            "terminal voltage": terminal_voltage,
        }
        self._variables.update(additional_variables)
