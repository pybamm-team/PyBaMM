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

    def __init__(self, current_scale, current_function):
        super().__init__()
        # Define current
        t = pybamm.t
        current = pybamm.standard_parameters.dimensionless_current(current_function, t)
        # Load reaction flux from submodels
        G = pybamm.interface.homogeneous_reaction(current)
        # Load diffusion model from submodels
        diffusion_model = pybamm.electrolyte.StefanMaxwellDiffusion(G)

        # Create own model from diffusion model
        self.update(diffusion_model)

        # Overwrite default solver for faster solution
        self.default_solver = pybamm.ScipySolver(method="BDF")
