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
        # Define concentration variable
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        c_e = pybamm.Variable("Concentration", whole_cell)
        # Load reaction flux from submodels
        j = pybamm.interface.homogeneous_reaction(whole_cell)
        # Load diffusion model from submodels
        diffusion_model = pybamm.electrolyte_diffusion.StefanMaxwell(c_e, j)

        # Create own model from diffusion model
        self.update(diffusion_model)
