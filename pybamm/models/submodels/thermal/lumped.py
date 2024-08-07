#
# Class for lumped thermal submodel
#
import pybamm

from .base_thermal import BaseThermal


class Lumped(BaseThermal):
    """
    Class for lumped thermal submodel. For more information see :footcite:t:`Timms2021`
    and :footcite:t:`Marquis2020`.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    options : dict, optional
        A dictionary of options to be passed to the model.

    """

    def __init__(self, param, options=None, x_average=False):
        super().__init__(param, options=options, x_average=x_average)
        pybamm.citations.register("Timms2021")

    def get_fundamental_variables(self):
        T_vol_av = pybamm.Variable(
            "Volume-averaged cell temperature [K]",
            scale=self.param.T_ref,
            print_name="T_av",
        )
        T_x_av = pybamm.PrimaryBroadcast(T_vol_av, ["current collector"])
        T_dict = {
            "negative current collector": T_x_av,
            "positive current collector": T_x_av,
            "x-averaged cell": T_x_av,
            "volume-averaged cell": T_vol_av,
        }
        for domain in ["negative electrode", "separator", "positive electrode"]:
            T_dict[domain] = pybamm.PrimaryBroadcast(T_x_av, domain)

        variables = self._get_standard_fundamental_variables(T_dict)

        return variables

    def get_coupled_variables(self, variables):
        variables.update(self._get_standard_coupled_variables(variables))

        # Newton cooling, accounting for surface area to volume ratio
        T_vol_av = variables["Volume-averaged cell temperature [K]"]
        T_amb = variables["Volume-averaged ambient temperature [K]"]
        V = variables["Cell thermal volume [m3]"]
        Q_cool_W = -self.param.h_total * (T_vol_av - T_amb) * self.param.A_cooling
        Q_cool_vol_av = Q_cool_W / V

        # Contact resistance heating Q_cr
        if self.options["contact resistance"] == "true":
            I = variables["Current [A]"]
            Q_cr_W = self.calculate_Q_cr_W(I, self.param.R_contact)
            V = self.param.V_cell
            Q_cr_vol_av = self.calculate_Q_cr_vol_av(I, self.param.R_contact, V)
        else:
            Q_cr_W = pybamm.Scalar(0)
            Q_cr_vol_av = Q_cr_W

        variables.update(
            {
                # Lumped cooling
                "Lumped total cooling [W.m-3]": Q_cool_vol_av,
                "Lumped total cooling [W]": Q_cool_W,
                # Contact resistance
                "Lumped contact resistance heating [W.m-3]": Q_cr_vol_av,
                "Lumped contact resistance heating [W]": Q_cr_W,
            }
        )
        return variables

    def set_rhs(self, variables):
        T_vol_av = variables["Volume-averaged cell temperature [K]"]
        Q_vol_av = variables["Volume-averaged total heating [W.m-3]"]
        Q_cool_vol_av = variables["Lumped total cooling [W.m-3]"]
        Q_cr_vol_av = variables["Lumped contact resistance heating [W.m-3]"]
        rho_c_p_eff_av = variables[
            "Volume-averaged effective heat capacity [J.K-1.m-3]"
        ]

        self.rhs = {T_vol_av: (Q_vol_av + Q_cr_vol_av + Q_cool_vol_av) / rho_c_p_eff_av}

    def set_initial_conditions(self, variables):
        T_vol_av = variables["Volume-averaged cell temperature [K]"]
        self.initial_conditions = {T_vol_av: self.param.T_init}

    def calculate_Q_cr_W(self, current, contact_resistance):
        Q_cr_W = current**2 * contact_resistance
        return Q_cr_W

    def calculate_Q_cr_vol_av(self, current, contact_resistance, volume):
        Q_cr_W = self.calculate_Q_cr_W(current, contact_resistance)
        Q_cr_vol_av = Q_cr_W / volume
        return Q_cr_vol_av
