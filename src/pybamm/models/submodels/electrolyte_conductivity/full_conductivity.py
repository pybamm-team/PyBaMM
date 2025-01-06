#
# Class for electrolyte conductivity employing stefan-maxwell
#
import pybamm

from .base_electrolyte_conductivity import BaseElectrolyteConductivity


class Full(BaseElectrolyteConductivity):
    """Full model for conservation of charge in the electrolyte employing the
    Stefan-Maxwell constitutive equations. (Full refers to unreduced by
    asymptotic methods)

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    options : dict, optional
        A dictionary of options to be passed to the model.
    """

    def __init__(self, param, options=None):
        super().__init__(param, options=options)

    def build(self, submodels):
        phi_e_dict = {}
        variables = {}
        for domain in self.options.whole_cell_domains:
            Dom = domain.capitalize().split()[0]
            name = f"{Dom} electrolyte potential [V]"
            phi_e_k = pybamm.Variable(
                name,
                domain=domain,
                auxiliary_domains={"secondary": "current collector"},
                reference=-self.param.n.prim.U_init,
            )
            phi_e_k.print_name = f"phi_e_{domain[0]}"
            phi_e_dict[domain] = phi_e_k

        variables["Electrolyte potential [V]"] = pybamm.concatenation(
            *phi_e_dict.values()
        )

        variables.update(self._get_standard_potential_variables(phi_e_dict))
        if self.options.electrode_types["negative"] == "planar":
            domains = ["separator", "positive electrode"]
        else:
            domains = ["negative electrode", "separator", "positive electrode"]
        T = pybamm.CoupledVariable(
            "Cell temperature [K]",
            domains,
            auxiliary_domains={"secondary": "current collector"},
        )
        self.coupled_variables.update({T.name: T})
        tor = pybamm.CoupledVariable(
            "Electrolyte transport efficiency",
            domains,
            auxiliary_domains={"secondary": "current collector"},
        )
        self.coupled_variables.update({tor.name: tor})
        c_e = pybamm.CoupledVariable(
            "Electrolyte concentration [mol.m-3]",
            domains,
            auxiliary_domains={"secondary": "current collector"},
        )
        self.coupled_variables.update({c_e.name: c_e})
        phi_e = pybamm.CoupledVariable(
            "Electrolyte potential [V]",
            domains,
            auxiliary_domains={"secondary": "current collector"},
        )
        self.coupled_variables.update({phi_e.name: phi_e})

        i_e = (self.param.kappa_e(c_e, T) * tor) * (
            self.param.chiRT_over_Fc(c_e, T) * pybamm.grad(c_e) - pybamm.grad(phi_e)
        )

        # Override print_name
        i_e.print_name = "i_e"

        variables.update(self._get_standard_current_variables(i_e))
        variables.update(self._get_electrolyte_overpotentials(variables))
        phi_e = variables["Electrolyte potential [V]"]
        i_e = variables["Electrolyte current density [A.m-2]"]

        # Variable summing all of the interfacial current densities
        sum_a_j = pybamm.CoupledVariable(
            "Sum of volumetric interfacial current densities [A.m-3]",
            domains,
            auxiliary_domains={"secondary": "current collector"},
        )
        self.coupled_variables.update({sum_a_j.name: sum_a_j})

        # Override print_name
        sum_a_j.print_name = "aj"

        # multiply by Lx**2 to improve conditioning
        self.algebraic = {phi_e: self.param.L_x**2 * (pybamm.div(i_e) - sum_a_j)}
        self.initial_conditions = {phi_e: -self.param.n.prim.U_init}
        self.variables.update(variables)
        if self.options.electrode_types["negative"] == "planar":
            phi_e_ref = pybamm.CoupledVariable(
                "Lithium metal interface electrolyte potential [V]",
                "current collector",
            )
            self.coupled_variables.update({phi_e_ref.name: phi_e_ref})
            lbc = (phi_e_ref, "Dirichlet")
        else:
            lbc = (pybamm.Scalar(0), "Neumann")
        self.boundary_conditions = {
            phi_e: {"left": lbc, "right": (pybamm.Scalar(0), "Neumann")}
        }
