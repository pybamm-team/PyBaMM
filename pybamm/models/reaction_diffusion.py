#
# Reaction-diffusion model
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class ReactionDiffusionModel(pybamm.StandardBatteryBaseModel):
    """Reaction-diffusion model.

    **Extends**: :class:`pybamm.BaseModel`

    """

    def __init__(self):
        super().__init__()
        "-----------------------------------------------------------------------------"
        "Parameters"
        param = pybamm.standard_parameters_lithium_ion

        "-----------------------------------------------------------------------------"
        "Model Variables"

        c_e = pybamm.standard_variables.c_e

        "-----------------------------------------------------------------------------"
        "Submodels"

        # Interfacial current density
        int_curr_model = pybamm.interface.InterfacialCurrent(param)
        j_vars = int_curr_model.get_homogeneous_interfacial_current()
        self.variables = j_vars

        # Electrolyte concentration
        eleclyte_conc_model = pybamm.electrolyte_diffusion.StefanMaxwell(param)
        eleclyte_conc_model.set_differential_system(c_e, self.variables)
        self.update(eleclyte_conc_model)
