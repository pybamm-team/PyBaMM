#
# Class for lumped thermal submodel
#
import pybamm

from .base_thermal import BaseModel


class Lumped1D(BaseModel):
    """Class for lumped thermal submodel

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.thermal.BaseModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):

        T_av = pybamm.standard_variables.T_av
        T_n = pybamm.PrimaryBroadcast(T_av, ["negative electrode"])
        T_s = pybamm.PrimaryBroadcast(T_av, ["separator"])
        T_p = pybamm.PrimaryBroadcast(T_av, ["positive electrode"])
        T = pybamm.Concatenation(T_n, T_s, T_p)

        variables = self._get_standard_fundamental_variables(T)
        return variables

    def get_coupled_variables(self, variables):
        variables.update(self._get_standard_coupled_variables(variables))
        return variables

    def _flux_law(self, T):
        """Fast heat diffusion (temperature has no spatial dependence)"""
        q = pybamm.FullBroadcast(
            pybamm.Scalar(0),
            ["negative electrode", "separator", "positive electrode"],
            "current collector",
        )
        return q

    def _unpack(self, variables):
        T_av = variables["Volume-averaged cell temperature"]
        q = variables["Heat flux"]
        Q_av = variables["Volume-averaged total heating"]
        return T_av, q, Q_av

    def _yz_average(self, var):
        """In 1D volume-averaged quantities are equal to x-avergaed quantities"""
        return var

    def _boundary_length(self):
        """
        Returns the length of the boundary available for cooling in the y-z direction.
        In 1D this is zero.
        """
        return pybamm.Scalar(0)

    def set_rhs(self, variables):
        T_av, _, Q_av = self._unpack(variables)

        # Cooling from faces of current collectors
        # TO DO: check -- should this be /l ?
        surface_cooling = -2 * self.param.h / (self.param.delta ** 2) * T_av

        # Cooling from cell edges O(delta)
        edge_cooling = -self._boundary_length() * self.param.h / self.param.delta * T_av

        self.rhs = {
            T_av: (self.param.B * Q_av + surface_cooling + edge_cooling)
            / (self.param.C_th * self.param.rho)
        }


class Lumped1plus1D(Lumped1D):
    """Class for 1+1D lumped thermal submodel"""

    def __init__(self, param):
        super().__init__(param)

    def _yz_average(self, var):
        """Computes the y-z avergage by integration over z (no y-direction)"""
        z = pybamm.standard_spatial_vars.z
        var_av = pybamm.Integral(var, z) / self.param.l_z
        return var_av

    def _boundary_length(self):
        """
        Returns the length of the boundary available for cooling in the y-z direction.
        """
        return 2 * self.param.l_z


class Lumped2plus1D(Lumped1D):
    """Class for 2+1D lumped thermal submodel"""

    def __init__(self, param):
        super().__init__(param)

    def _yz_average(self, var):
        """Computes the y-z avergage by integration over y and z"""
        y = pybamm.standard_spatial_vars.y
        z = pybamm.standard_spatial_vars.z
        var_av = pybamm.Integral(var, [y, z]) / self.param.l_y / self.param.l_z
        return var_av

    def _boundary_length(self):
        """
        Returns the length of the boundary available for cooling in the y-z direction.
        """
        return 2 * (self.param.l_z + self.param.l_z)
