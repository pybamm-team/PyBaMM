#
# Class for irreversible lithium plating
#
import pybamm
from .base_plating import BasePlating


class IrreversiblePlating(BasePlating):
    """Base class for irreversible lithium plating.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain of the model either 'Negative' or 'Positive'

    References
    ----------
    .. [1] SEJ O'Kane, ID Campbell, MWJ Marzook, GJ Offer and M Marinescu. "Physical
           Origin of the Differential Voltage Minimum Associated with Li Plating in
           Lithium-Ion Batteries". Journal of The Electrochemical Society,
           167:090540, 2019

    **Extends:** :class:`pybamm.lithium_plating.BasePlating`
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)
        pybamm.citations.register("OKane2020")

    def get_fundamental_variables(self):
        c_plated_Li = pybamm.Variable(
            f"{self.domain.capitalize()} electrode lithium plating concentration",
            domain=self.domain.lower() + " electrode",
            auxiliary_domains={"secondary": "current collector"},
        )

        variables = self._get_standard_concentration_variables(c_plated_Li)

        return variables

    def get_coupled_variables(self, variables):
        param = self.param
        phi_s_n = variables[f"{self.domain} electrode potential"]
        phi_e_n = variables[f"{self.domain} electrolyte potential"]
        c_e_n = variables[f"{self.domain} electrolyte concentration"]
        eta_sei = variables[f"{self.domain} electrode SEI film overpotential"]
        C_plating = param.C_plating
        phi_ref = param.U_n_ref / param.potential_scale

        # need to revise for thermal case
        # j_stripping is always negative, because there is no stripping, only plating
        j_stripping = (
            -(1 / C_plating)
            * c_e_n
            * (pybamm.exp(-0.5 * (phi_s_n - phi_e_n + phi_ref + eta_sei)))
        )

        variables.update(self._get_standard_reaction_variables(j_stripping))

        # Update whole cell variables, which also updates the "sum of" variables
        if (
            "Negative electrode lithium plating interfacial current density"
            in variables
            and "Positive electrode lithium plating interfacial current density"
            in variables
            and "Lithium plating interfacial current density" not in variables
        ):
            variables.update(
                self._get_standard_whole_cell_interfacial_current_variables(variables)
            )

        return variables

    def set_rhs(self, variables):
        c_plated_Li = variables[
            f"{self.domain} electrode lithium plating concentration"
        ]
        j_stripping = variables[
            f"{self.domain} electrode lithium plating interfacial current density"
        ]
        Gamma_plating = self.param.Gamma_plating

        self.rhs = {c_plated_Li: -Gamma_plating * j_stripping}

    def set_initial_conditions(self, variables):
        c_plated_Li = variables[
            f"{self.domain} electrode lithium plating concentration"
        ]
        c_plated_Li_0 = self.param.c_plated_Li_0

        self.initial_conditions = {c_plated_Li: c_plated_Li_0}
