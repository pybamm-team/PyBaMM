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

    def __init__(self, param, options=None, x_average=False, **kwargs):
        super().__init__(param, options=options, x_average=x_average, **kwargs)

    def build(self):
        # Set the x-averaged temperature to the ambient temperature, which can be
        # specified as a function of space (y, z) only and time
        y = pybamm.standard_spatial_vars.y
        z = pybamm.standard_spatial_vars.z
        # Broadcast t to be the same size as y and z (to catch cases where the ambient
        # temperature is a function of time only)
        t_broadcast = pybamm.PrimaryBroadcast(pybamm.t, "current collector")
        T_x_av = self.param.T_amb(y, z, t_broadcast)
        T_vol_av = self._yz_average(T_x_av)
        T_dict = self._get_x_uniform_temperatures(T_x_av, T_vol_av)
        self.variables.update(self._get_standard_fundamental_variables(T_dict))

        if self.options["calculate heat source for isothermal models"] == "true":
            self.variables.update(self._get_heat_source_variables(T_dict))
        else:
            zero = pybamm.Scalar(0)
            for var in [
                "Ohmic heating [W.m-3]",
                "X-averaged Ohmic heating [W.m-3]",
                "Volume-averaged Ohmic heating [W.m-3]",
                "Ohmic heating per unit electrode-pair area [W.m-2]",
                "Ohmic heating [W]",
                "Irreversible electrochemical heating [W.m-3]",
                "X-averaged irreversible electrochemical heating [W.m-3]",
                "Volume-averaged irreversible electrochemical heating [W.m-3]",
                "Irreversible electrochemical heating per unit electrode-pair area [W.m-2]",
                "Irreversible electrochemical heating [W]",
                "Reversible heating [W.m-3]",
                "X-averaged reversible heating [W.m-3]",
                "Volume-averaged reversible heating [W.m-3]",
                "Reversible heating per unit electrode-pair area [W.m-2]",
                "Reversible heating [W]",
                "Total heating [W.m-3]",
                "X-averaged total heating [W.m-3]",
                "Volume-averaged total heating [W.m-3]",
                "Total heating per unit electrode-pair area [W.m-2]",
                "Total heating [W]",
                "Negative current collector Ohmic heating [W.m-3]",
                "Positive current collector Ohmic heating [W.m-3]",
                "Surface total cooling [W.m-3]",
                "Surface total cooling [W]",
            ]:
                self.variables[var] = zero
