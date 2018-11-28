#
# Reaction-diffusion model
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class ReactionDiffusionModel(pybamm.BaseModel):
    """Reaction-diffusion model.

    Parameters
    ----------
    tests : dict, optional
        A dictionary for testing the convergence of the numerical solution:
            * {} (default): We are not running in test mode, use built-ins.
            * {'inits': dict of initial conditions,
               'bcs': dict of boundary conditions,
               'sources': dict of source terms
               }: To be used for testing convergence to an exact solution.
    """

    def __init__(self, tests={}):
        super().__init__(tests)
        self.name = "Reaction Diffusion"

    @property
    def submodels(self):
        if not self.simulation_set:
            raise ValueError("Simulation is not set")
        return [
            pybamm.submodels.ElectrolyteTransport(
                self.param.electrolyte,
                self.operators["xc"],
                self.mesh.whole,
                self.tests,
            )
        ]
