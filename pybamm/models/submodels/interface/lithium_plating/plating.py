#
# Class for lithium plating
#
import pybamm
from .base_plating import BasePlating


class Plating(BasePlating):
    """Class for lithium plating.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    x_average : bool
        Whether to use x-averaged variables (SPM, SPMe, etc) or full variables (DFN)
    options : dict, optional
        A dictionary of options to be passed to the model.

    References
    ----------
    .. [1] SEJ O'Kane, ID Campbell, MWJ Marzook, GJ Offer and M Marinescu. "Physical
           Origin of the Differential Voltage Minimum Associated with Li Plating in
           Lithium-Ion Batteries". Journal of The Electrochemical Society,
           167:090540, 2020
    .. [2] SEJ O'Kane, W Ai, G Madabattula, D Alonso-Alvarez, R Timms, V Sulzer,
           JS Edge, B Wu, GJ Offer and M Marinescu. "Lithium-ion battery degradation:
           how to model it". Physical Chemistry: Chemical Physics, 24:7909, 2022
    """

    def __init__(self, param, x_average, options):
        super().__init__(param, options)
        self.x_average = x_average
        pybamm.citations.register("OKane2020")
        pybamm.citations.register("OKane2022")

    def get_fundamental_variables(self):
        if self.x_average is True:
            c_plated_Li_av = pybamm.Variable(
                "X-averaged lithium plating concentration [mol.m-3]",
                domain="current collector",
                scale=self.param.c_Li_typ,
            )
            c_plated_Li = pybamm.PrimaryBroadcast(c_plated_Li_av, "negative electrode")
            c_dead_Li_av = pybamm.Variable(
                "X-averaged dead lithium concentration [mol.m-3]",
                domain="current collector",
            )
            c_dead_Li = pybamm.PrimaryBroadcast(c_dead_Li_av, "negative electrode")
        else:
            c_plated_Li = pybamm.Variable(
                "Lithium plating concentration [mol.m-3]",
                domain="negative electrode",
                auxiliary_domains={"secondary": "current collector"},
                scale=self.param.c_Li_typ,
            )
            c_dead_Li = pybamm.Variable(
                "Dead lithium concentration [mol.m-3]",
                domain="negative electrode",
                auxiliary_domains={"secondary": "current collector"},
            )

        variables = self._get_standard_concentration_variables(c_plated_Li, c_dead_Li)

        return variables

    def get_coupled_variables(self, variables):
        param = self.param
        delta_phi = variables["Negative electrode surface potential difference [V]"]
        c_e_n = variables["Negative electrolyte concentration [mol.m-3]"]
        T = variables["Negative electrode temperature [K]"]
        eta_sei = variables["SEI film overpotential [V]"]
        c_plated_Li = variables["Lithium plating concentration [mol.m-3]"]
        j0_stripping = param.j0_stripping(c_e_n, c_plated_Li, T)
        j0_plating = param.j0_plating(c_e_n, c_plated_Li, T)

        eta_stripping = delta_phi + eta_sei
        eta_plating = -eta_stripping
        F_RT = param.F / (param.R * T)
        # NEW: transfer coefficients can be set by the user
        alpha_stripping = self.param.alpha_stripping
        alpha_plating = self.param.alpha_plating

        if self.options["lithium plating"] in ["reversible", "partially reversible"]:
            j_stripping = j0_stripping * pybamm.exp(
                F_RT * alpha_stripping * eta_stripping
            ) - j0_plating * pybamm.exp(F_RT * alpha_plating * eta_plating)
        elif self.options["lithium plating"] == "irreversible":
            # j_stripping is always negative, because there is no stripping, only
            # plating
            j_stripping = -j0_plating * pybamm.exp(F_RT * alpha_plating * eta_plating)

        variables.update(self._get_standard_overpotential_variables(eta_stripping))
        variables.update(self._get_standard_reaction_variables(j_stripping))

        # Update whole cell variables, which also updates the "sum of" variables
        variables.update(super().get_coupled_variables(variables))

        return variables

    def set_rhs(self, variables):
        if self.x_average is True:
            c_plated_Li = variables[
                "X-averaged lithium plating concentration [mol.m-3]"
            ]
            c_dead_Li = variables["X-averaged dead lithium concentration [mol.m-3]"]
            a_j_stripping = variables[
                "X-averaged lithium plating volumetric "
                "interfacial current density [A.m-3]"
            ]
            L_sei = variables["X-averaged total SEI thickness [m]"]
        else:
            c_plated_Li = variables["Lithium plating concentration [mol.m-3]"]
            c_dead_Li = variables["Dead lithium concentration [mol.m-3]"]
            a_j_stripping = variables[
                "Lithium plating volumetric interfacial current density [A.m-3]"
            ]
            L_sei = variables["Total SEI thickness [m]"]

        # In the partially reversible plating model, coupling term turns reversible
        # lithium into dead lithium. In other plating models, it is zero.
        if self.options["lithium plating"] == "partially reversible":
            dead_lithium_decay_rate = self.param.dead_lithium_decay_rate(L_sei)
            coupling_term = dead_lithium_decay_rate * c_plated_Li
        else:
            coupling_term = pybamm.Scalar(0)

        self.rhs = {
            c_plated_Li: -a_j_stripping / self.param.F - coupling_term,
            c_dead_Li: coupling_term,
        }

    def set_initial_conditions(self, variables):
        if self.x_average is True:
            c_plated_Li = variables[
                "X-averaged lithium plating concentration [mol.m-3]"
            ]
            c_dead_Li = variables["X-averaged dead lithium concentration [mol.m-3]"]
        else:
            c_plated_Li = variables["Lithium plating concentration [mol.m-3]"]
            c_dead_Li = variables["Dead lithium concentration [mol.m-3]"]
        c_plated_Li_0 = self.param.c_plated_Li_0
        zero = pybamm.Scalar(0)

        self.initial_conditions = {c_plated_Li: c_plated_Li_0, c_dead_Li: zero}
