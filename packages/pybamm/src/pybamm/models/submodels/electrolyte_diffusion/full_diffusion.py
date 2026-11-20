#
# Class for electrolyte diffusion employing stefan-maxwell
#
import numpy as np

import pybamm

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
    x_average_state : bool, optional
        Whether to make the x-averaged electrolyte concentration an algebraic state
        when a surface form is used. This keeps the Jacobian sparse for the
        composite conductivity submodels, which read the x-average in every
        equation, but only adds an extra state for the full conductivity submodel,
        which never reads it. Default is False.
    """

    def __init__(self, param, options=None, x_average_state=False):
        super().__init__(param, options)
        self.x_average_state = x_average_state

    @property
    def _c_e_av_is_state(self) -> bool:
        return self.x_average_state and self.options["surface form"] != "false"

    def _get_standard_whole_cell_concentration_variables(self, c_e):
        if not self._c_e_av_is_state:
            return super()._get_standard_whole_cell_concentration_variables(c_e)

        c_e.print_name = "c_e"
        c_e_av = pybamm.Variable(
            "X-averaged electrolyte concentration [mol.m-3]",
            domain="current collector",
            bounds=(0, np.inf),
            scale=self.param.c_e_init_av,
        )
        variables = {
            "Electrolyte concentration [mol.m-3]": c_e,
            "X-averaged electrolyte concentration [mol.m-3]": c_e_av,
        }
        nondim = {
            name.replace("[mol.m-3]", "[Molar]"): var / 1000
            for name, var in variables.items()
        }
        variables.update(nondim)

        return variables

    def build(self):
        whole_cell = self.options.whole_cell_domains
        cell_domains = {"primary": whole_cell, "secondary": "current collector"}

        def coupled(name):
            return self.coupled_variable(name, domains=cell_domains)

        tor = coupled("Electrolyte transport efficiency")
        i_e = coupled("Electrolyte current density [A.m-2]")
        v_box = coupled("Volume-averaged velocity [m.s-1]")
        div_Vbox = coupled("Transverse volume-averaged acceleration [m.s-2]")
        T = coupled("Cell temperature [K]")
        sum_s_a_j = coupled("Sum of electrolyte reaction source terms [A.m-3]")
        i_boundary_cc = self.coupled_variable(
            "Current collector current density [A.m-2]", domain="current collector"
        )

        eps_c_e_dict = {}
        for domain in whole_cell:
            eps_c_e_k = pybamm.Variable(
                f"{domain.capitalize()} porosity times concentration [mol.m-3]",
                domain=domain,
                auxiliary_domains={"secondary": "current collector"},
                bounds=(0, np.inf),
                scale=self.param.c_e_init_av,
            )
            eps_c_e_k.print_name = f"eps_c_e_{domain[0]}"
            eps_c_e_dict[domain] = eps_c_e_k

        variables = self._get_standard_porosity_times_concentration_variables(
            eps_c_e_dict
        )
        variables.update(
            self._get_concentration_from_porosity_times_concentration(eps_c_e_dict)
        )
        eps_c_e = variables["Porosity times concentration [mol.m-3]"]
        c_e = variables["Electrolyte concentration [mol.m-3]"]
        c_e_conc = variables["Electrolyte concentration concatenation [mol.m-3]"]

        N_e_diffusion = -tor * self.param.D_e(c_e, T) * pybamm.grad(c_e)
        N_e_migration = self.param.t_plus(c_e, T) * i_e / self.param.F
        N_e_convection = c_e * v_box
        N_e = N_e_diffusion + N_e_migration + N_e_convection

        variables.update(self._get_standard_flux_variables(N_e))
        variables.update(
            {
                "Electrolyte diffusion flux [mol.m-2.s-1]": N_e_diffusion,
                "Electrolyte migration flux [mol.m-2.s-1]": N_e_migration,
                "Electrolyte convection flux [mol.m-2.s-1]": N_e_convection,
            }
        )
        self.variables.update(variables)

        source_terms = sum_s_a_j / self.param.F
        self.rhs = {eps_c_e: -pybamm.div(N_e) + source_terms - c_e * div_Vbox}
        self.initial_conditions = {
            eps_c_e: self.param.epsilon_init * self.param.c_e_init
        }

        if self._c_e_av_is_state:
            c_e_av = variables["X-averaged electrolyte concentration [mol.m-3]"]
            self.algebraic = {
                c_e_av: (c_e_av - pybamm.x_average(c_e)) / self.param.c_e_init_av
            }
            self.initial_conditions[c_e_av] = self.param.c_e_init

        if whole_cell[0] == "negative electrode":
            # left bc at anode/current collector interface
            lbc = pybamm.Scalar(0)
        else:
            # left bc at lithium metal/separator interface, assuming v_box = 0
            lbc = (
                pybamm.boundary_value(
                    -(1 - self.param.t_plus(c_e, T))
                    / (tor * self.param.D_e(c_e, T) * self.param.F),
                    "left",
                )
                * i_boundary_cc
            )
        bcs = {"left": (lbc, "Neumann"), "right": (pybamm.Scalar(0), "Neumann")}
        # add boundary conditions to both forms of the concentration
        self.boundary_conditions = {c_e: bcs, c_e_conc: bcs}
