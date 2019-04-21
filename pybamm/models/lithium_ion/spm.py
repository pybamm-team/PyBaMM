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

        c_s_n = pybamm.standard_variables.c_s_n
        c_s_p = pybamm.standard_variables.c_s_p

        "-----------------------------------------------------------------------------"
        "Submodels"

        # Interfacial current density
        int_curr_model = pybamm.interface.InterfacialCurrent(param)
        j_vars = int_curr_model.get_homogeneous_interfacial_current()
        self.variables.update(j_vars)

        # Particle models
        j_n = j_vars["Negative electrode interfacial current density"].orphans[0]
        negative_particle_model = pybamm.particle.Standard(param)
        negative_particle_model.set_differential_system(c_s_n, j_n, broadcast=True)
        j_p = j_vars["Positive electrode interfacial current density"].orphans[0]
        positive_particle_model = pybamm.particle.Standard(param)
        positive_particle_model.set_differential_system(c_s_p, j_p, broadcast=True)
        self.update(negative_particle_model, positive_particle_model)

        "-----------------------------------------------------------------------------"
        "Post-Processing"

        # Electrolyte concentration
        electrolyte_conc_model = pybamm.electrolyte_diffusion.StefanMaxwell(param)
        conc_vars = electrolyte_conc_model.get_constant_concentration_variables()
        self.variables.update(conc_vars)

        # Exchange-current density
        ecd_vars = int_curr_model.get_exchange_current_densities(self.variables)
        self.variables.update(ecd_vars)

        # Potentials
        pot_model = pybamm.potential.Potential(param)
        c_s_n_surf = self.variables["Negative particle surface concentration"]
        c_s_p_surf = self.variables["Positive particle surface concentration"]
        ocp_vars = pot_model.get_open_circuit_potentials(c_s_n_surf, c_s_p_surf)
        eta_r_vars = pot_model.get_reaction_overpotentials(self.variables, "current")
        self.variables.update({**ocp_vars, **eta_r_vars})

        # Electrolyte current
        eleclyte_current_model = pybamm.electrolyte_current.MacInnesStefanMaxwell(param)
        elyte_vars = eleclyte_current_model.get_explicit_leading_order(self.variables)
        self.variables.update(elyte_vars)

        # Electrode
        electrode_model = pybamm.electrode.Ohm(param)
        electrode_vars = electrode_model.get_explicit_leading_order(self.variables)
        self.variables.update(electrode_vars)

        "-----------------------------------------------------------------------------"
        "Defaults and Solver Conditions"
        # default geometry
        self.default_geometry = pybamm.Geometry("1D macro", "1D micro")
