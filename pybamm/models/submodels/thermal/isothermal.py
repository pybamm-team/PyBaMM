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
    """

    def __init__(self, param, options=None):
        super().__init__(param, options=options)

    def get_fundamental_variables(self):
        T_amb = self.param.T_amb(pybamm.t)
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
                "Ohmic heating [W.m-3]",
                "X-averaged Ohmic heating [W.m-3]",
                "Volume-averaged Ohmic heating [W.m-3]",
                "Irreversible electrochemical heating [W.m-3]",
                "X-averaged irreversible electrochemical heating [W.m-3]",
                "Volume-averaged irreversible electrochemical heating [W.m-3]",
                "Reversible heating [W.m-3]",
                "X-averaged reversible heating [W.m-3]",
                "Volume-averaged reversible heating [W.m-3]",
                "Total heating [W.m-3]",
                "X-averaged total heating [W.m-3]",
                "Volume-averaged total heating [W.m-3]",
            ]:
                # All variables are zero
                variables.update({var: zero})

        return variables
