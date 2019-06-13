#
# Class for constant porosity
#
import pybamm

from .base_porosity import BaseModel


class Constant(BaseModel):
    """Base class for constant porosity

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`pybamm.BaseSubModel`
    """

    def get_fundamental_variables(self):

        eps_n_av = self.param.epsilon_n
        eps_s_av = self.param.epsilon_s
        eps_p_av = self.param.epsilon_p

        eps_n = pybamm.Broadcast(eps_n_av, ["negative electrode"])
        eps_s = pybamm.Broadcast(eps_s_av, ["separator"])
        eps_p = pybamm.Broadcast(eps_p_av, ["positive electrode"])
        eps = pybamm.Concatenation(eps_n, eps_s, eps_p)

        zero = pybamm.Scalar(0)
        deps_n_dt = pybamm.Broadcast(zero, ["negative electrode"])
        deps_s_dt = pybamm.Broadcast(zero, ["separator"])
        deps_p_dt = pybamm.Broadcast(zero, ["positive electrode"])
        deps_dt = pybamm.Concatenation(deps_n_dt, deps_s_dt, deps_p_dt)

        variables = self._get_standard_porosity_variables(eps)
        variables.update(self._get_standard_porosity_change_variables(deps_dt))

        return variables
