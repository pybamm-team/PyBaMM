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
           167:090540, 2019

    **Extends:** :class:`pybamm.lithium_plating.BasePlating`
    """

    def __init__(self, param, x_average, options):
        super().__init__(param, options)
        self.x_average = x_average
        pybamm.citations.register("OKane2020")

    def get_fundamental_variables(self):
        if self.x_average is True:
            c_plated_Li_av = pybamm.Variable(
                "X-averaged lithium plating concentration",
                domain="current collector",
            )
            c_plated_Li = pybamm.PrimaryBroadcast(c_plated_Li_av, "negative electrode")
        else:
            c_plated_Li = pybamm.Variable(
                "Lithium plating concentration",
                domain="negative electrode",
                auxiliary_domains={"secondary": "current collector"},
            )

        variables = self._get_standard_concentration_variables(c_plated_Li)

        return variables

    def get_coupled_variables(self, variables):
        param = self.param
        delta_phi = variables["Negative electrode surface potential difference"]
        c_e_n = variables["Negative electrolyte concentration"]
        T = variables["Negative electrode temperature"]
        eta_sei = variables["SEI film overpotential"]
        c_plated_Li = variables["Lithium plating concentration"]
        j0_stripping = param.j0_stripping(c_e_n, c_plated_Li, T)
        j0_plating = param.j0_plating(c_e_n, c_plated_Li, T)
        phi_ref = param.U_n_ref / param.potential_scale

        eta_stripping = delta_phi + phi_ref + eta_sei
        eta_plating = -eta_stripping
        prefactor = 1 / (2 * (1 + self.param.Theta * T))

        if self.options["lithium plating"] == "reversible":
            j_stripping = j0_stripping * pybamm.exp(
                prefactor * eta_stripping
            ) - j0_plating * pybamm.exp(prefactor * eta_plating)
        elif self.options["lithium plating"] == "irreversible":
            # j_stripping is always negative, because there is no stripping, only
            # plating
            j_stripping = -j0_plating * pybamm.exp(prefactor * eta_plating)

        variables.update(self._get_standard_overpotential_variables(eta_stripping))
        variables.update(self._get_standard_reaction_variables(j_stripping))

        # Update whole cell variables, which also updates the "sum of" variables
        variables.update(super().get_coupled_variables(variables))

        return variables

    def set_rhs(self, variables):
        if self.x_average is True:
            c_plated_Li = variables["X-averaged lithium plating concentration"]
            j_stripping = variables[
                "X-averaged lithium plating interfacial current density"
            ]
            # Note a is dimensionless (has a constant value of 1 if the surface
            # area does not change)
            a = variables["X-averaged negative electrode surface area to volume ratio"]
        else:
            c_plated_Li = variables["Lithium plating concentration"]
            j_stripping = variables["Lithium plating interfacial current density"]
            a = variables["Negative electrode surface area to volume ratio"]

        Gamma_plating = self.param.Gamma_plating

        self.rhs = {c_plated_Li: -Gamma_plating * a * j_stripping}

    def set_initial_conditions(self, variables):
        if self.x_average is True:
            c_plated_Li = variables["X-averaged lithium plating concentration"]
        else:
            c_plated_Li = variables["Lithium plating concentration"]
        c_plated_Li_0 = self.param.c_plated_Li_0

        self.initial_conditions = {c_plated_Li: c_plated_Li_0}
