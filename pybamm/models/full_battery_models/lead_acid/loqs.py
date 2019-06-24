#
# Lead-acid LOQS model
#
import pybamm
from .base_lead_acid_model import BaseModel


class LOQS(BaseModel):
    """Leading-Order Quasi-Static model for lead-acid, from [1]_.

    References
    ----------
    .. [1] V Sulzer, SJ Chapman, CP Please, DA Howey, and CW Monroe. Faster Lead-Acid
           Battery Simulations from Porous-Electrode Theory: II. Asymptotic Analysis.
           arXiv preprint arXiv:1902.01774, 2019.

    **Extends:** :class:`pybamm.BaseLeadAcidModel`
    """

    def __init__(self, options=None):
        super().__init__(options)
        self.name = "LOQS model"
        self.use_jacobian = False

        self.set_reactions()
        self.set_current_collector_submodel()
        self.set_interfacial_submodel()
        self.set_convection_submodel()
        self.set_porosity_submodel()
        self.set_negative_electrode_submodel()
        self.set_electrolyte_submodel()
        self.set_positive_electrode_submodel()
        self.set_thermal_submodel()

        self.build_model()

    def set_reactions(self):

        # Should probably refactor as this is a bit clunky at the moment
        # Maybe each reaction as a Reaction class so we can just list names of classes
        self.reactions = {
            "main": {
                "neg": {
                    "s_plus": self.param.s_n,
                    "j": "Average negative electrode interfacial current density",
                },
                "pos": {
                    "s_plus": self.param.s_p,
                    "j": "Average positive electrode interfacial current density",
                },
            }
        }

    def set_current_collector_submodel(self):

        self.submodels["current collector"] = pybamm.current_collector.Uniform(
            self.param, "Negative"
        )

    def set_porosity_submodel(self):

        self.submodels["porosity"] = pybamm.porosity.LeadingOrder(self.param)

    def set_convection_submodel(self):

        if self.options["convection"] is False:
            self.submodels["convection"] = pybamm.convection.NoConvection(self.param)
        if self.options["convection"] is True:
            self.submodels["convection"] = pybamm.convection.LeadingOrder(self.param)

    def set_interfacial_submodel(self):

        self.submodels[
            "negative interface"
        ] = pybamm.interface.inverse_butler_volmer.LeadAcid(self.param, "Negative")
        self.submodels[
            "positive interface"
        ] = pybamm.interface.inverse_butler_volmer.LeadAcid(self.param, "Positive")

    def set_negative_electrode_submodel(self):

        self.submodels["negative electrode"] = pybamm.electrode.ohm.Leading(
            self.param, "Negative"
        )

    def set_positive_electrode_submodel(self):

        self.submodels["positive electrode"] = pybamm.electrode.ohm.Leading(
            self.param, "Positive"
        )

    def set_electrolyte_submodel(self):

        electrolyte = pybamm.electrolyte.stefan_maxwell

        self.submodels[
            "electrolyte conductivity"
        ] = electrolyte.conductivity.LeadingOrderModel(self.param)

        self.submodels[
            "electrolyte diffusion"
        ] = electrolyte.diffusion.LeadingOrderModel(
            self.param, self.reactions, ocp=True
        )

    #     self.set_model_variables()
    #     self.set_interface_and_electrolyte_submodels()
    #     self.set_porosity_submodel()
    #     self.set_diffusion_submodel()
    #     self.set_current_variables()
    #     self.set_convection_variables()

    #     # ODEs only (don't use jacobian, use base spatial method)
    #     self.use_jacobian = False

    # def set_model_variables(self):
    #     "Set variables for the model"
    #     if self.options["bc_options"]["dimensionality"] == 0:
    #         curr_coll_domain = []
    #     elif self.options["bc_options"]["dimensionality"] == 1:
    #         curr_coll_domain = ["current collector"]

    #     c_e = pybamm.Variable("Electrolyte concentration", curr_coll_domain)

    #     # Piecewise constant epsilon
    #     eps_n_pc = pybamm.Variable("Negative electrode porosity", curr_coll_domain)
    #     eps_s_pc = pybamm.Variable("Separator porosity", curr_coll_domain)
    #     eps_p_pc = pybamm.Variable("Positive electrode porosity", curr_coll_domain)

    #     epsilon = pybamm.Concatenation(
    #         pybamm.Broadcast(eps_n_pc, ["negative electrode"]),
    #         pybamm.Broadcast(eps_s_pc, ["separator"]),
    #         pybamm.Broadcast(eps_p_pc, ["positive electrode"]),
    #     )

    #     self.variables.update({"Electrolyte concentration": c_e, "Porosity": epsilon})

    #     if self.options["capacitance"] is not False:
    #         delta_phi_n = pybamm.Variable(
    #             "Negative electrode surface potential difference", curr_coll_domain
    #         )
    #         delta_phi_p = pybamm.Variable(
    #             "Positive electrode surface potential difference", curr_coll_domain
    #         )
    #         self.variables.update(
    #             {
    #                 "Negative electrode surface potential difference": delta_phi_n,
    #                 "Positive electrode surface potential difference": delta_phi_p,
    #             }
    #         )

    # def set_boundary_conditions(self, bc_variables=None):
    #     "Set boundary conditions, dependent on self.options"
    #     # TODO: edit to allow constant-current and constant-power control
    #     param = self.set_of_parameters
    #     dimensionality = self.options["bc_options"]["dimensionality"]
    #     if dimensionality == 0:
    #         current_bc = param.current_with_time
    #         self.variables["Current collector current density"] = current_bc
    #     elif dimensionality == 1:
    #         current_collector_model = pybamm.vertical.Vertical(param)
    #         current_collector_model.set_leading_order_vertical_current(bc_variables)
    #         self.update(current_collector_model)

    # def set_interface_and_electrolyte_submodels(self):
    #     param = self.set_of_parameters
    #     c_e = self.variables["Electrolyte concentration"]
    #     # Exchange-current density
    #     neg = ["negative electrode"]
    #     pos = ["positive electrode"]
    #     int_curr_model = pybamm.interface.LeadAcidReaction(param)
    #     j0_n = int_curr_model.get_exchange_current_densities(c_e, neg)
    #     j0_p = int_curr_model.get_exchange_current_densities(c_e, pos)

    #     # Open-circuit potential
    #     ocp_n = param.U_n(c_e)
    #     ocp_p = param.U_p(c_e)

    #     if self.options["capacitance"] is not False:
    #         delta_phi_n = self.variables[
    #             "Negative electrode surface potential difference"
    #         ]
    #         delta_phi_p = self.variables[
    #             "Positive electrode surface potential difference"
    #         ]
    #         self.set_boundary_conditions(self.variables)

    #         # Reaction overpotential
    #         eta_r_n = delta_phi_n - ocp_n
    #         eta_r_p = delta_phi_p - ocp_p

    #         # Interfacial current density
    #         j_n = int_curr_model.get_butler_volmer(j0_n, eta_r_n, neg)
    #         j_p = int_curr_model.get_butler_volmer(j0_p, eta_r_p, pos)
    #         self.reactions = {
    #             "main": {
    #                 "neg": {"s_plus": param.s_n, "aj": j_n},
    #                 "pos": {"s_plus": param.s_p, "aj": j_p},
    #             }
    #         }

    #         # Electrolyte current
    #         eleclyte_current_model = pybamm.electrolyte_current.MacInnesCapacitance(
    #             param, self.options["capacitance"]
    #         )
    #         eleclyte_current_model.set_leading_order_system(
    #             self.variables, self.reactions, neg
    #         )
    #         eleclyte_current_model.set_leading_order_system(
    #             self.variables, self.reactions, pos
    #         )
    #         self.update(eleclyte_current_model)

    #     else:
    #         i_boundary_cc = param.current_with_time
    #         self.variables["Current collector current density"] = i_boundary_cc

    #         # Interfacial current density
    #         j_n = int_curr_model.get_homogeneous_interfacial_current(i_boundary_cc,
    # neg)
    #         j_p = int_curr_model.get_homogeneous_interfacial_current(i_boundary_cc,
    # pos)
    #         self.reactions = {
    #             "main": {
    #                 "neg": {"s_plus": param.s_n, "aj": j_n},
    #                 "pos": {"s_plus": param.s_p, "aj": j_p},
    #             }
    #         }

    #         # Potentials
    #         eta_r_n = int_curr_model.get_inverse_butler_volmer(j_n, j0_n, neg)
    #         eta_r_p = int_curr_model.get_inverse_butler_volmer(j_p, j0_p, pos)

    #     # Exchange-current density
    #     j_vars = int_curr_model.get_derived_interfacial_currents(j_n, j_p, j0_n, j0_p)
    #     self.variables.update(j_vars)

    #     # Potentials
    #     pot_model = pybamm.potential.Potential(param)
    #     ocp_vars = pot_model.get_derived_open_circuit_potentials(ocp_n, ocp_p)
    #     eta_r_vars = pot_model.get_derived_reaction_overpotentials(eta_r_n, eta_r_p)
    #     self.variables.update({**ocp_vars, **eta_r_vars})

    # def set_porosity_submodel(self):
    #     param = self.set_of_parameters
    #     porosity_model = pybamm.porosity.Standard(param)
    #     porosity_model.set_leading_order_system(self.variables)
    #     self.update(porosity_model)

    #     # Update reactions
    #     deps_n_dt = self.variables["Negative electrode porosity change"].orphans[0]
    #     deps_p_dt = self.variables["Positive electrode porosity change"].orphans[0]
    #     self.reactions["main"]["neg"]["deps_dt"] = deps_n_dt
    #     self.reactions["main"]["pos"]["deps_dt"] = deps_p_dt

    # def set_diffusion_submodel(self):
    #     param = self.set_of_parameters
    #     eleclyte_conc_model = pybamm.electrolyte_diffusion.StefanMaxwell(param)
    #     eleclyte_conc_model.set_leading_order_system(self.variables, self.reactions)
    #     self.update(eleclyte_conc_model)

    # def set_current_variables(self):
    #     param = self.set_of_parameters

    #     elyte_postproc_model = pybamm.electrolyte_current.ElectrolyteCurrentBaseModel(
    #         param
    #     )
    #     # Electrolyte current
    #     elyte_vars = elyte_postproc_model.get_explicit_leading_order(self.variables)
    #     self.variables.update(elyte_vars)

    #     # Electrode
    #     electrode_model = pybamm.electrode.Ohm(param)
    #     electrode_vars = electrode_model.get_explicit_leading_order(self.variables)
    #     self.variables.update(electrode_vars)

    #     # Cut-off voltage
    #     # Hack to extract voltage at the tabs in 2D
    #     voltage = self.variables["Terminal voltage"]
    #     if self.options["bc_options"]["dimensionality"] == 1:
    #         voltage.domain = "current collector"
    #         voltage = pybamm.boundary_value(voltage, "right")
    #     self.events.append(voltage - param.voltage_low_cut)

    # def set_convection_variables(self):
    #     velocity_model = pybamm.velocity.Velocity(self.set_of_parameters)
    #     if self.options["convection"] is not False:
    #         velocity_vars = velocity_model.get_explicit_leading_order(self.variables)
    #         self.variables.update(velocity_vars)
    #     else:
    #         whole_cell = ["negative electrode", "separator", "positive electrode"]
    #         v_box = pybamm.Broadcast(0, whole_cell)
    #         dVbox_dz = pybamm.Broadcast(0, whole_cell)
    #         self.variables.update(velocity_model.get_variables(v_box, dVbox_dz))

    @property
    def default_spatial_methods(self):
        # ODEs only in the macroscale, so use base spatial method
        return {
            "macroscale": pybamm.FiniteVolume,
            "current collector": pybamm.FiniteVolume,
        }

    @property
    def default_geometry(self):
        if self.options["bc_options"]["dimensionality"] == 0:
            return pybamm.Geometry("1D macro")
        elif self.options["bc_options"]["dimensionality"] == 1:
            return pybamm.Geometry("1+1D macro")

    @property
    def default_solver(self):
        """
        Create and return the default solver for this model
        """
        # Different solver depending on whether we solve ODEs or DAEs
        if self.options["capacitance"] == "algebraic":
            return pybamm.ScikitsDaeSolver()
        else:
            return pybamm.ScikitsOdeSolver()
