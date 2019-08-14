#
# Class for lumped thermal submodel which accounts for current collectors
#
import pybamm

from .base_current_collector_thermal import BaseModel


class Lumped1D(BaseModel):
    """Class for 1D lumped thermal submodel which accounts for current collectors.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.thermal.current_collector.BaseModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):

        T_av = pybamm.standard_variables.T_av

        variables = self._get_standard_fundamental_variables(T_av)
        return variables

    def get_coupled_variables(self, variables):
        variables.update(self._get_standard_coupled_variables(variables))
        return variables

    def _unpack(self, variables):
        T_volume_av = variables["Volume-averaged cell temperature"]
        q = variables["Heat flux"]
        Q_volume_av = variables["Volume-averaged total heating"]
        return T_volume_av, q, Q_volume_av

    def _current_collector_heating(self, variables):
        """Returns the heat source terms in the 0D current collector"""
        i_boundary_cc = variables["Current collector current density"]
        Q_s_cn = i_boundary_cc ** 2 / self.param.sigma_cn
        Q_s_cp = i_boundary_cc ** 2 / self.param.sigma_cp
        return Q_s_cn, Q_s_cp

    def _surface_cooling_coefficient(self):
        """Returns the surface cooling coefficient in 1D"""
        return -2 * self.param.h / (self.param.delta ** 2)

    def _yz_average(self, var):
        """In 1D volume-averaged quantities are unchanged"""
        return var

    def set_rhs(self, variables):
        T_volume_av, _, Q_volume_av = self._unpack(variables)

        cooling_coeff = self._surface_cooling_coefficient()

        self.rhs = {
            T_volume_av: (self.param.B * Q_volume_av + cooling_coeff * T_volume_av)
            / self.param.C_th
        }


class LumpedNplus1D(Lumped1D):
    """Base class for N+1D lumped thermal submodels"""

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):

        T_volume_av = pybamm.standard_variables.T_volume_av
        T_av = pybamm.PrimaryBroadcast(T_volume_av, ["current collector"])

        variables = self._get_standard_fundamental_variables(T_av)
        return variables


class Lumped1plus1D(LumpedNplus1D):
    """Class for 1+1D lumped thermal submodel"""

    def __init__(self, param):
        super().__init__(param)

    def _current_collector_heating(self, variables):
        """Returns the heat source terms in the 1D current collector"""
        # TODO: implement grad to calculate actual heating instead of average
        # approximate heating
        i_boundary_cc = variables["Current collector current density"]
        Q_s_cn = i_boundary_cc ** 2 / self.param.sigma_cn
        Q_s_cp = i_boundary_cc ** 2 / self.param.sigma_cp
        return Q_s_cn, Q_s_cp

    def _surface_cooling_coefficient(self):
        """Returns the surface cooling coefficient in 1+1D"""
        return (
            -2 * self.param.h / (self.param.delta ** 2)
            - 2 * self.param.l_z * self.param.h / self.param.delta
        )

    def _yz_average(self, var):
        """Computes the y-z avergage by integration over z (no y-direction)"""
        return pybamm.z_average(var)


class Lumped2plus1D(LumpedNplus1D):
    """Class for 2+1D lumped thermal submodel"""

    def __init__(self, param):
        super().__init__(param)

    def _current_collector_heating(self, variables):
        """Returns the heat source terms in the 2D current collector"""
        # TODO: implement grad to calculate actual heating instead of average
        # approximate heating
        i_boundary_cc = variables["Current collector current density"]
        Q_s_cn = i_boundary_cc ** 2 / self.param.sigma_cn
        Q_s_cp = i_boundary_cc ** 2 / self.param.sigma_cp
        return Q_s_cn, Q_s_cp

    def _surface_cooling_coefficient(self):
        """Returns the surface cooling coefficient in 2+1D"""
        return (
            -2 * self.param.h / (self.param.delta ** 2)
            - 2 * (self.param.l_y + self.param.l_z) * self.param.h / self.param.delta
        )

    def _yz_average(self, var):
        """Computes the y-z avergage by integration over y and z"""
        return pybamm.yz_average(var)
