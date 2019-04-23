#
# Lead-acid LOQS model
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class LOQS(pybamm.LeadAcidBaseModel):
    """Leading-Order Quasi-Static model for lead-acid.

    **Extends**: :class:`pybamm.LeadAcidBaseModel`

    """

    def __init__(self):
        super().__init__()

        "-----------------------------------------------------------------------------"
        "Parameters"
        param = pybamm.standard_parameters_lead_acid

        "-----------------------------------------------------------------------------"
        "Model Variables"

        c_e = pybamm.Variable("Electrolyte concentration")
        eps = pybamm.standard_variables.eps_piecewise_constant

        "-----------------------------------------------------------------------------"
        "Submodels"

        # Interfacial current density
        int_curr_model = pybamm.interface.InterfacialCurrent(param)
        j_vars = int_curr_model.get_homogeneous_interfacial_current()
        self.variables.update(j_vars)

        # Porosity
        j = j_vars["Interfacial current density"]
        porosity_model = pybamm.porosity.Standard(param)
        porosity_model.set_leading_order_system(eps, j)
        self.update(porosity_model)

        # Electrolyte concentration
        eleclyte_conc_model = pybamm.electrolyte_diffusion.StefanMaxwell(param)
        eleclyte_conc_model.set_leading_order_system(c_e, self.variables)
        self.update(eleclyte_conc_model)

        "-----------------------------------------------------------------------------"
        "Post-Processing"

        # Exchange-current density
        ecd_vars = int_curr_model.get_exchange_current_densities(
            self.variables, intercalation=False
        )
        self.variables.update(ecd_vars)

        # Potentials
        pot_model = pybamm.potential.Potential(param)
        c_e_n = self.variables["Negative electrode electrolyte concentration"]
        c_e_p = self.variables["Positive electrode electrolyte concentration"]
        ocp_vars = pot_model.get_open_circuit_potentials(c_e_n, c_e_p)
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
