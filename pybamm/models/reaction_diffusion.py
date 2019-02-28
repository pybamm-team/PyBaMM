#
# Reaction-diffusion model
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class ReactionDiffusionModel(pybamm.BaseModel):
    """Reaction-diffusion model.

    **Extends**: :class:`pybamm.BaseModel`

    """

    def __init__(self):
        super().__init__()
        # Define current
        current = pybamm.standard_parameters.current_with_time
        # Load reaction flux from submodels
        G = pybamm.interface.homogeneous_reaction(current)
        # Load diffusion model from submodels
        diffusion_model = pybamm.electrolyte.StefanMaxwellDiffusion(G)

        # Create own model from diffusion model
        self.update(diffusion_model)

        # Overwrite default solver for faster solution
        self.default_solver = pybamm.ScipySolver(method="BDF")
