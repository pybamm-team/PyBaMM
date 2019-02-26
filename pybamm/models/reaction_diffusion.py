#
# Reaction-diffusion model
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class ReactionDiffusionModel(pybamm.BaseModel):
    """Reaction-diffusion model.

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
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        c_e = pybamm.Variable("c_e", whole_cell)

        "Model Parameters and functions"
        #

        "Interface Conditions"
        G = pybamm.interface.homogeneous_reaction(whole_cell)

        "Model Equations"
        self.update(pybamm.electrolyte_diffusion.StefanMaxwell(c_e, G))

        "Additional Conditions"
        additional_bcs = {}
        self._boundary_conditions.update(additional_bcs)

        "Additional Model Variables"
        additional_variables = {}
        self._variables.update(additional_variables)
