#
# Single Particle Model (SPM)
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import numpy as np


class SPM(pybamm.LithiumIonBaseModel):
    """Single Particle Model (SPM) of a lithium-ion battery.
    **Extends:** :class:`pybamm.LithiumIonBaseModel`
    """

    def __init__(self):
        super().__init__()

        "-----------------------------------------------------------------------------"
        "Parameters"
        param = pybamm.standard_parameters_lithium_ion

        "-----------------------------------------------------------------------------"
        "Model Variables"

        # Particle concentration
        c_s_n = pybamm.Variable(
            "Negative particle concentration", domain="negative particle"
        )
        c_s_p = pybamm.Variable(
            "Positive particle concentration", domain="positive particle"
        )

        "-----------------------------------------------------------------------------"
        "Submodels"

        # Interfacial current density
        int_curr_model = pybamm.interface.InterfacialCurrent(param)
        int_curr_model.set_homogeneous_interfacial_current()
        self.update(int_curr_model)

        # Particle models
        j_n = int_curr_model.variables["Negative electrode interfacial current density"]
        j_p = int_curr_model.variables["Positive electrode interfacial current density"]
        negative_particle_model = pybamm.particle.Standard(c_s_n, j_n, param)
        positive_particle_model = pybamm.particle.Standard(c_s_p, j_p, param)
        self.update(negative_particle_model, positive_particle_model)

        "-----------------------------------------------------------------------------"
        "Post-Processing"

        # Electrolyte concentration
        eleclyte_conc_model = pybamm.electrolyte_diffusion.ConstantConcentration(param)
        self.update(eleclyte_conc_model)

        # Exchange-current density
        int_curr_model.set_exchange_current_densities(self.variables)
        self.update(int_curr_model)

        # Potentials
        potential_model = pybamm.potential.Potential(param)
        potential_model.set_open_circuit_potentials(
            pybamm.Broadcast(pybamm.surf(c_s_n), ["negative electrode"]),
            pybamm.Broadcast(pybamm.surf(c_s_p), ["positive electrode"]),
        )
        potential_model.set_reaction_overpotentials(self.variables, "current")
        self.update(potential_model)

        # Electrolyte current
        eleclyte_current_model = pybamm.electrolyte_current.MacInnesStefanMaxwell(param)
        eleclyte_current_model.set_explicit_leading_order(self.variables)
        self.update(eleclyte_current_model)

        # Electrode
        electrode_model = pybamm.electrode.Ohm(param)
        electrode_model.explicit_leading_order(self.variables)
        self.update(electrode_model)

        "-----------------------------------------------------------------------------"
        "Defaults and Solver Conditions"
        # default geometry
        self.default_geometry = pybamm.Geometry("1D macro", "1D micro")

        # Cut-off if either concentration goes negative
        self.events = [pybamm.Function(np.min, c_s_n), pybamm.Function(np.min, c_s_p)]
