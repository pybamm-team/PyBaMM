#
# Class for isothermal submodel
#
import pybamm

from .base_thermal import BaseThermal


class Isothermal(BaseThermal):
    """
    Class for isothermal submodel.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    options : dict, optional
        A dictionary of options to be passed to the model.

    **Extends:** :class:`pybamm.thermal.BaseThermal`
    """

    def __init__(self, param, options=None):
        super().__init__(param, options=options)

    def get_fundamental_variables(self):
        T_amb = self.param.T_amb(pybamm.t * self.param.timescale)
        T_x_av = pybamm.PrimaryBroadcast(T_amb, "current collector")

        T_dict = {
            "negative current collector": T_x_av,
            "positive current collector": T_x_av,
            "x-averaged cell": T_x_av,
            "volume-averaged cell": T_x_av,
        }
        for domain in ["negative electrode", "separator", "positive electrode"]:
            T_dict[domain] = pybamm.PrimaryBroadcast(T_x_av, domain)

        variables = self._get_standard_fundamental_variables(T_dict)

        return variables

    def get_coupled_variables(self, variables):
        if self.options["calculate heat source for isothermal models"] == "true":
            variables.update(self._get_standard_coupled_variables(variables))
        else:
            zero = pybamm.Scalar(0)
            for var in [
                "Ohmic heating",
                "X-averaged Ohmic heating",
                "Volume-averaged Ohmic heating",
                "Irreversible electrochemical heating",
                "X-averaged irreversible electrochemical heating",
                "Volume-averaged irreversible electrochemical heating",
                "Reversible heating",
                "X-averaged reversible heating",
                "Volume-averaged reversible heating",
                "Total heating",
                "X-averaged total heating",
                "Volume-averaged total heating",
            ]:
                # Both dimensionless and dimensional variable are zero
                variables.update({var: zero, f"{var} [W.m-3]": zero})

        return variables
