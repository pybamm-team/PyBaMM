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
        self.variables = {}

        "-----------------------------------------------------------------------------"
        "Model Variables"

        c_e = pybamm.standard_variables.c_e

        "-----------------------------------------------------------------------------"
        "Submodels"

        # Interfacial current density
        int_curr_model = pybamm.interface.InterfacialCurrent(param)
        j_n, j_p = int_curr_model.get_homogeneous_interfacial_current()

        # Electrolyte concentration
        j_n = pybamm.Broadcast(j_n, ["negative electrode"])
        j_p = pybamm.Broadcast(j_p, ["positive electrode"])
        reactions = {
            "main": {"neg": {"s_plus": 1, "aj": j_n}, "pos": {"s_plus": 1, "aj": j_p}}
        }
        eleclyte_conc_model = pybamm.electrolyte_diffusion.StefanMaxwell(param)
        eleclyte_conc_model.set_differential_system(c_e, reactions)
        self.update(eleclyte_conc_model)
