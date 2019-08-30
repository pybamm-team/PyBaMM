#
# Class for lumped thermal submodel
#
from .base_thermal import BaseModel


class NoCurrentCollector(BaseModel):
    """Class for x-lumped thermal submodel without current collectors

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.thermal.BaseModel`
    """

    def __init__(self, param):
        super().__init__(param)

    # def _unpack(self, variables):
    #     T_av = variables["X-averaged cell temperature"]
    #     q = variables["Heat flux"]
    #     Q_av = variables["X-averaged total heating"]
    #     return T_av, q, Q_av

    def set_rhs(self, variables):
        T_av = variables["X-averaged cell temperature"]
        Q_av = variables["X-averaged total heating"]

        self.rhs = {
            T_av: (
                self.param.B * Q_av - 2 * self.param.h / (self.param.delta ** 2) * T_av
            )
            / (self.param.C_th * self.param.rho)
        }
