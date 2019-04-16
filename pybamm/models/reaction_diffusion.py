#
# Reaction-diffusion model
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
import autograd.numpy as np


class ReactionDiffusionModel(pybamm.BaseModel):
    """Reaction-diffusion model.

    **Extends**: :class:`pybamm.BaseModel`

    """

    def __init__(self):
        super().__init__()
        # Parameters
        param = pybamm.standard_parameters_lithium_ion

        # Variables and parameters
        #
        # Define concentration variable
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        c_e = pybamm.Variable("Concentration", whole_cell)

        #
        # Submodels
        #
        # Use cos(x) as the source term
        x = pybamm.SpatialVariable("x", domain=whole_cell)
        j = pybamm.Function(np.cos, x)
        # Use uniform porosity
        epsilon = pybamm.Scalar(1)
        # Load diffusion model from submodels
        diffusion_model = pybamm.electrolyte_diffusion.StefanMaxwell(
            c_e, j, param, epsilon=epsilon
        )

        # Create own model from diffusion model
        self.update(diffusion_model)

        # Add j to variables dict
        self.variables.update({"Interfacial current density": j})
