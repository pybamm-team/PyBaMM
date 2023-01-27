#
# Class for electrolyte diffusion employing stefan-maxwell
#
import pybamm
import numpy as np
from .base_electrolyte_diffusion import BaseElectrolyteDiffusion


class Full(BaseElectrolyteDiffusion):
    """Class for conservation of mass in the electrolyte employing the
    Stefan-Maxwell constitutive equations. (Full refers to unreduced by
    asymptotic methods)

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    options : dict, optional
        A dictionary of options to be passed to the model.

    **Extends:** :class:`pybamm.electrolyte_diffusion.BaseElectrolyteDiffusion`
    """

    def __init__(self, param, options=None):
        super().__init__(param, options)

    def get_fundamental_variables(self):
        eps_c_e_dict = {}
        for domain in self.options.whole_cell_domains:
            Domain = domain.capitalize()
            eps_c_e_k = pybamm.Variable(
                f"{Domain} porosity times concentration",
                domain=domain,
                auxiliary_domains={"secondary": "current collector"},
                bounds=(0, np.inf),
            )
            eps_c_e_k.print_name = f"eps_c_e_{domain[0]}"
            eps_c_e_dict[domain] = eps_c_e_k

        variables = self._get_standard_porosity_times_concentration_variables(
            eps_c_e_dict
        )

        return variables

    def get_coupled_variables(self, variables):

        c_e_dict = {}
        for domain in self.options.whole_cell_domains:
            Domain = domain.capitalize()
            eps_k = variables[f"{Domain} porosity"]
            eps_c_e_k = variables[f"{Domain} porosity times concentration"]
            c_e_k = eps_c_e_k / eps_k
            c_e_dict[domain] = c_e_k

        variables["Electrolyte concentration concatenation"] = pybamm.concatenation(
            *c_e_dict.values()
        )
        variables.update(self._get_standard_domain_concentration_variables(c_e_dict))

        c_e = variables["Porosity times concentration"] / variables["Porosity"]
        variables.update(self._get_standard_whole_cell_concentration_variables(c_e))

        # Whole domain
        tor = variables["Electrolyte transport efficiency"]
        i_e = variables["Electrolyte current density"]
        v_box = variables["Volume-averaged velocity"]
        T = variables["Cell temperature"]

        param = self.param

        N_e_diffusion = -tor * param.D_e(c_e, T) * pybamm.grad(c_e)
        N_e_migration = param.C_e * param.t_plus(c_e, T) * i_e / param.gamma_e
        N_e_convection = param.C_e * c_e * v_box

        N_e = N_e_diffusion + N_e_migration + N_e_convection

        variables.update(self._get_standard_flux_variables(N_e))
        variables.update(
            {
                "Electrolyte diffusion flux": N_e_diffusion,
                "Electrolyte migration flux": N_e_migration,
                "Electrolyte convection flux": N_e_convection,
            }
        )

        return variables

    def set_rhs(self, variables):

        param = self.param

        eps_c_e = variables["Porosity times concentration"]
        c_e = variables["Electrolyte concentration"]
        T = variables["Cell temperature"]
        N_e_diffusion = variables["Electrolyte diffusion flux"]
        N_e_convection = variables["Electrolyte convection flux"]
        N_e = N_e_diffusion + N_e_convection
        div_Vbox = variables["Transverse volume-averaged acceleration"]

        sum_s_j = variables["Sum of electrolyte reaction source terms"]
        sum_a_j = variables["Sum of volumetric interfacial current densities"]
        sum_s_j.print_name = "a"
        source_terms = (sum_s_j - param.t_plus(c_e, T) * sum_a_j) / self.param.gamma_e

        self.rhs = {
            eps_c_e: -pybamm.div(N_e) / param.C_e + source_terms - c_e * div_Vbox
        }

    def set_initial_conditions(self, variables):

        eps_c_e = variables["Porosity times concentration"]

        self.initial_conditions = {
            eps_c_e: self.param.epsilon_init * self.param.c_e_init
        }

    def set_boundary_conditions(self, variables):
        param = self.param
        c_e = variables["Electrolyte concentration"]
        c_e_conc = variables["Electrolyte concentration concatenation"]
        T = variables["Cell temperature"]
        tor = variables["Electrolyte transport efficiency"]
        i_boundary_cc = variables["Current collector current density"]

        def flux_bc(side):
            # returns the flux at a separator/electrode interface
            # assuming v_box = 0 for now
            return (
                pybamm.boundary_value(
                    -(1 - param.t_plus(c_e, T))
                    / (tor * param.gamma_e * param.D_e(c_e, T)),
                    side,
                )
                * i_boundary_cc
                * param.C_e
            )

        if self.options.whole_cell_domains[0] == "negative electrode":
            # left bc at anode/current collector interface
            lbc = pybamm.Scalar(0)
        elif self.options.whole_cell_domains[0] == "separator":
            # left bc at anode/separator interface
            lbc = flux_bc("left")
        if self.options.whole_cell_domains[-1] == "positive electrode":
            # right bc at cathode/current collector interface
            rbc = pybamm.Scalar(0)
        # elif self.options.whole_cell_domains[-1] == "separator":
        #     # right bc at separator/cathode interface
        #     rbc = flux_bc("right")

        # add boundary conditions to both forms of the concentration
        self.boundary_conditions = {
            c_e: {"left": (lbc, "Neumann"), "right": (rbc, "Neumann")},
            c_e_conc: {"left": (lbc, "Neumann"), "right": (rbc, "Neumann")},
        }
